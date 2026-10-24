#!/usr/bin/env python3
"""
Camber (GitHub Student Pack: L4 24GB, 5 GPU h/month) dispatcher. Runs on GitHub Actions.
  python3 worker/camber_dispatch.py probe                      # discover CLI/engines/sizes, run a tiny GPU+internet test job
  python3 worker/camber_dispatch.py run --project P --stage S [--only a,b] [--engine E --size Z]
Env: CAMBER_API_KEY. Uploads worker/gpu/ + a job spec to the stash, submits `camber job create --gpu`, waits, pulls logs + outputs.
"""
import os, sys, json, time, shutil, pathlib, argparse, subprocess, tarfile, re
ROOT = pathlib.Path(__file__).resolve().parents[1]
QR = ROOT / "quality_reports"

def sh(cmd, check=False, timeout=900):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT, input="y\n"*20)
    out = (r.stdout + ("\n" + r.stderr if r.stderr.strip() else "")).strip()
    print(f"$ {' '.join(cmd)}\n{out[:4000]}\n", flush=True)
    if check and r.returncode != 0: raise SystemExit(f"command failed: {' '.join(cmd)}")
    return r.returncode, out

def stash_root():
    rc, out = sh(["camber","me"]); m = re.search(r"Stash:\s*(stash://\S+)", out)
    return (m.group(1) if m else "stash://").rstrip("/")
def probe(rep):
    rc, out = subprocess.run(["camber","app","describe","gromacs-protein-md","--output","json"], capture_output=True, text=True, cwd=ROOT).returncode, subprocess.run(["camber","app","describe","gromacs-protein-md","--output","json"], capture_output=True, text=True, cwd=ROOT).stdout
    try:
        d = json.loads(out[out.index("{"):]); print("GROMACS command:", d.get("command")); print("GROMACS spec:", json.dumps(d.get("spec"), indent=1)[:3000]); print("GROMACS job_config:", json.dumps(d.get("job_config"))[:1500])
    except Exception as e: print("describe parse failed", e, out[:500])
    root = stash_root(); sp = f"{root}/room313"
    sh(["camber","stash","cp",str(ROOT/"worker"/"gpu"/"probe.sh"),f"{sp}/probe.sh"])
    sh(["camber","agent","list"])
    msg = f"Run this exact shell command on the compute node and do nothing else: cd {sp.replace(root+'/','~/')} 2>/dev/null || cd /stash/room313 2>/dev/null || cd $(find / -maxdepth 4 -type d -name room313 2>/dev/null | head -1); bash probe.sh > probe_out.txt 2>&1; tail -60 probe_out.txt. Reply with the full output of tail."
    me = root.replace("stash://","").strip("/")
    sh(["camber","agent","create","--help"])
    rc, out = sh(["camber","agent","create","--name","room313","--alias","room313","--description","Room 313 GPU runner: executes shell commands on the node exactly as given","--output","json"])
    if rc != 0: rc, out = sh(["camber","agent","create","room313","--output","json"])
    sh(["camber","agent","list"])
    tag = f"@{me}.room313"
    rc, out = sh(["camber","chat","-a",tag,"-m","Reply with exactly: PONG","--output","json","--no-thinking"], timeout=300)
    if rc == 0:
        rc, out = sh(["camber","chat","-a",tag,"-m",msg,"--output","json","--no-thinking"], timeout=1500); (QR/"chat_probe.json").write_text(out)
    sh(["camber","stash","ls",sp])
    rc, out = sh(["camber","stash","cp",f"{sp}/probe_out.txt",str(QR/"probe_out.txt")])
    if (QR/"probe_out.txt").exists(): print("=== probe_out.txt ===\n" + (QR/"probe_out.txt").read_text()[:5000])
    if os.environ.get("PROBE_HELP_ONLY"): return

def submit(cmd_str, engine, size, path, gpu=True, extra=()):
    c = ["camber","job","create","--engine",engine,"--size",size,"--cmd",cmd_str,"--path",path] + (["--gpu"] if gpu else []) + list(extra)
    rc, out = sh(c)
    m = re.search(r"\b(\d{3,})\b", out)   # job id
    return (m.group(1) if m else None), out

def wait(job_id, poll=30, timeout_min=360):
    t0 = time.time(); last = ""
    while time.time() - t0 < timeout_min*60:
        rc, out = sh(["camber","job","get",job_id])
        try: st = json.loads(out[out.index("{"):]).get("status") or json.loads(out[out.index("{"):]).get("state")
        except Exception: st = re.search(r"(COMPLETED|FAILED|RUNNING|PENDING|QUEUED|CANCELLED|ERROR|SUCCEEDED)", out, re.I); st = st.group(1) if st else "?"
        if st != last: print(f"  {int(time.time()-t0)}s status={st}", flush=True); last = st
        if str(st).upper() in ("COMPLETED","FAILED","CANCELLED","ERROR","SUCCEEDED","FINISHED"): return st
        time.sleep(poll)
    return "TIMEOUT"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["probe","run","collect"]); ap.add_argument("--project"); ap.add_argument("--stage", default="video"); ap.add_argument("--only", default="")
    ap.add_argument("--engine", default=os.environ.get("CAMBER_ENGINE") or "base"); ap.add_argument("--size", default=os.environ.get("CAMBER_SIZE") or "xsmall"); ap.add_argument("--stash-path", default=os.environ.get("CAMBER_STASH") or "room313")
    a = ap.parse_args()
    if not os.environ.get("CAMBER_API_KEY"): raise SystemExit("CAMBER_API_KEY missing")
    ts = int(time.time()); outdir = QR / f"camber_{a.mode}_{ts}"; outdir.mkdir(parents=True, exist_ok=True); rep = []
    if a.mode == "probe":
        probe(rep)
        if os.environ.get("PROBE_HELP_ONLY"): (outdir/"probe_report.json").write_text(json.dumps(rep, indent=1)); return 0
        # tiny GPU + internet test job via stash
        root = stash_root(); sp = f"{root}/{a.stash_path}"; test = ROOT / "worker" / "gpu" / "probe.sh"
        sh(["camber","stash","mkdir",sp]); rc, out = sh(["camber","stash","cp",str(test),f"{sp}/probe.sh"]); rep.append({"cmd":"stash cp probe.sh","rc":rc,"out":out})
        rc, out = sh(["camber","stash","ls",sp]); rep.append({"cmd":"stash ls","out":out})
        jid, out = submit("bash probe.sh", a.engine, a.size, sp + "/", gpu=True); rep.append({"cmd":"job create probe","out":out,"job":jid})
        if jid:
            st = wait(jid, timeout_min=20); rep.append({"job":jid,"final":st})
            rc, out = sh(["camber","job","logs",jid]); (outdir/f"job_{jid}.log").write_text(out); rep.append({"cmd":"job logs","out":out[:6000]})
            rc, out = sh(["camber","stash","ls",sp]); rep.append({"cmd":"stash ls after","out":out})
        (outdir/"probe_report.json").write_text(json.dumps(rep, indent=1)); print("probe report ->", outdir); return 0
    if a.mode == "collect":
        root = stash_root(); sp = f"{root}/{a.stash_path}"; proj = ROOT/"projects"/a.project; gen = proj/"generated"
        return collect(sp, outdir, proj, gen, "collect")
    # ---- run (agent route): package worker + manifest + assets -> stash; agent starts gpu/run.sh in background; poll run.log in stash; collect out.tar.gz
    proj = ROOT/"projects"/a.project; pkg = outdir/"pkg"; pkg.mkdir()
    shutil.copytree(ROOT/"worker"/"gpu", pkg/"gpu"); shutil.copy(proj/"manifest.json", pkg/"manifest.json")
    gen = proj/"generated"
    if gen.exists(): shutil.copytree(gen, pkg/"generated", ignore=shutil.ignore_patterns("*.mp4", "*.wav", "*.mp3"))   # keyframes only (i2v inputs)
    (pkg/"job.json").write_text(json.dumps({"project": a.project, "stage": a.stage, "only": [x for x in a.only.split(",") if x]}))
    tar = outdir/"pkg.tar.gz"
    with tarfile.open(tar, "w:gz") as t: t.add(pkg, arcname="pkg")
    root = stash_root(); sp = f"{root}/{a.stash_path}"; me = root.replace("stash://","").strip("/"); tag = f"@{me}.room313"
    sh(["camber","stash","mkdir",sp]); sh(["camber","stash","rm",f"{sp}/pkg.tar.gz"]); sh(["camber","stash","rm",f"{sp}/run.log"])
    sh(["camber","stash","cp",str(tar),f"{sp}/pkg.tar.gz"], check=True)
    cmd = f"cd ~/{a.stash_path} && rm -rf pkg && tar xzf pkg.tar.gz && cd pkg && mkdir -p out && (nohup bash gpu/run.sh > ../run.log 2>&1 &) ; sleep 8; echo STARTED; tail -5 ../run.log"
    msg = f"Run this exact shell command on the compute node, do not modify it, do not run anything else, then reply with its output only:\n{cmd}"
    rc, out = sh(["camber","chat","-a",tag,"-m",msg,"--output","json","--no-thinking"], timeout=600); (outdir/"chat_start.json").write_text(out)
    if "STARTED" not in out: print("agent did not confirm start"); 
    # poll run.log
    t0 = time.time(); last = ""; final = "TIMEOUT"
    while time.time() - t0 < a_timeout(a):
        time.sleep(60); lf = outdir/"run.log"
        r = subprocess.run(["camber","stash","cp",f"{sp}/run.log",str(lf)], capture_output=True, text=True, cwd=ROOT)
        if lf.exists():
            txt = lf.read_text(errors="ignore"); tail = "\n".join(txt.strip().splitlines()[-3:])
            if tail != last: print(f"  {int(time.time()-t0)}s | {tail[-300:]}", flush=True); last = tail
            if "RUN_DONE" in txt: final = "DONE"; break
            if "RUN_FAILED" in txt: final = "FAILED"; break
    print("final:", final)
    rc = collect(sp, outdir, proj, gen, final)
    shutil.rmtree(pkg, ignore_errors=True); tar.unlink(missing_ok=True); return rc

def collect(sp, outdir, proj, gen, final):
    rc, out = sh(["camber","stash","cp",f"{sp}/pkg/out.tar.gz",str(outdir/"out.tar.gz")])
    o = outdir/"out.tar.gz"; n = 0
    if o.exists():
        with tarfile.open(o) as t: t.extractall(outdir)
        for f in (outdir/"out"/"generated").rglob("*"):
            if f.is_file(): d = gen/f.relative_to(outdir/"out"/"generated"); d.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, d); n += 1
        stp = proj/"status.json"; status = json.loads(stp.read_text()) if stp.exists() else {}
        ks = outdir/"out"/"gpu_status.json"
        if ks.exists():
            for k, v in json.loads(ks.read_text()).items(): status[k] = {**v, "worker": "camber-L4"}
            stp.write_text(json.dumps(status, indent=2))
        wl = outdir/"out"/"worker.log"
        if wl.exists(): shutil.copy(wl, outdir/"worker.log")
        shutil.rmtree(outdir/"out", ignore_errors=True); o.unlink()
    print(f"{final}; merged {n} files"); return 0 if n else 1

def a_timeout(a): return int(os.environ.get("CAMBER_TIMEOUT_MIN", "240")) * 60

if __name__ == "__main__": sys.exit(main())

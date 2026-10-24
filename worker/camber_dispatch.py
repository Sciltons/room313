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
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    out = (r.stdout + ("\n" + r.stderr if r.stderr.strip() else "")).strip()
    print(f"$ {' '.join(cmd)}\n{out[:4000]}\n", flush=True)
    if check and r.returncode != 0: raise SystemExit(f"command failed: {' '.join(cmd)}")
    return r.returncode, out

def probe(rep):
    for c in [["camber","version"],["camber","me"],["camber","--help"],["camber","job","--help"],["camber","job","create","--help"],
              ["camber","stash","--help"],["camber","engine","--help"],["camber","engine","list"],["camber","node","--help"]]:
        rc, out = sh(c); rep.append({"cmd": " ".join(c), "rc": rc, "out": out})

def submit(cmd_str, engine, size, path, gpu=True, extra=()):
    c = ["camber","job","create","--engine",engine,"--size",size,"--cmd",cmd_str,"--path",path] + (["--gpu"] if gpu else []) + list(extra)
    rc, out = sh(c)
    m = re.search(r"\b(\d{3,})\b", out)   # job id
    return (m.group(1) if m else None), out

def wait(job_id, poll=30, timeout_min=360):
    t0 = time.time(); last = ""
    while time.time() - t0 < timeout_min*60:
        rc, out = sh(["camber","job","get",job_id,"--output","json"])
        try: st = json.loads(out[out.index("{"):]).get("status") or json.loads(out[out.index("{"):]).get("state")
        except Exception: st = re.search(r"(COMPLETED|FAILED|RUNNING|PENDING|QUEUED|CANCELLED|ERROR|SUCCEEDED)", out, re.I); st = st.group(1) if st else "?"
        if st != last: print(f"  {int(time.time()-t0)}s status={st}", flush=True); last = st
        if str(st).upper() in ("COMPLETED","FAILED","CANCELLED","ERROR","SUCCEEDED","FINISHED"): return st
        time.sleep(poll)
    return "TIMEOUT"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["probe","run"]); ap.add_argument("--project"); ap.add_argument("--stage", default="video"); ap.add_argument("--only", default="")
    ap.add_argument("--engine", default=os.environ.get("CAMBER_ENGINE","mpi")); ap.add_argument("--size", default=os.environ.get("CAMBER_SIZE","SMALL")); ap.add_argument("--stash-path", default=os.environ.get("CAMBER_STASH","room313"))
    a = ap.parse_args()
    if not os.environ.get("CAMBER_API_KEY"): raise SystemExit("CAMBER_API_KEY missing")
    ts = int(time.time()); outdir = QR / f"camber_{a.mode}_{ts}"; outdir.mkdir(parents=True, exist_ok=True); rep = []
    if a.mode == "probe":
        probe(rep)
        # tiny GPU + internet test job via stash
        test = ROOT / "worker" / "gpu" / "probe.sh"
        rc, out = sh(["camber","stash","cp",str(test),f"stash://{a.stash_path}/probe.sh"]); rep.append({"cmd":"stash cp probe.sh","rc":rc,"out":out})
        if rc != 0:
            rc, out = sh(["camber","stash","upload",str(test),f"{a.stash_path}/"]); rep.append({"cmd":"stash upload probe.sh","rc":rc,"out":out})
        jid, out = submit("bash probe.sh", a.engine, a.size, a.stash_path, gpu=True); rep.append({"cmd":"job create probe","out":out,"job":jid})
        if jid:
            st = wait(jid, timeout_min=20); rep.append({"job":jid,"final":st})
            sh(["camber","job","log",jid,str(outdir)]); rc, out = sh(["camber","stash","ls",a.stash_path]); rep.append({"cmd":"stash ls","out":out})
        (outdir/"probe_report.json").write_text(json.dumps(rep, indent=1)); print("probe report ->", outdir); return 0
    # ---- run: package worker + manifest + needed assets, submit, wait, collect
    proj = ROOT/"projects"/a.project; pkg = outdir/"pkg"; pkg.mkdir()
    shutil.copytree(ROOT/"worker"/"gpu", pkg/"gpu"); shutil.copy(proj/"manifest.json", pkg/"manifest.json")
    gen = proj/"generated"
    if gen.exists(): shutil.copytree(gen, pkg/"generated")   # keyframes needed for i2v (small)
    (pkg/"job.json").write_text(json.dumps({"project": a.project, "stage": a.stage, "only": [x for x in a.only.split(",") if x]}))
    tar = outdir/"pkg.tar.gz"
    with tarfile.open(tar, "w:gz") as t: t.add(pkg, arcname="pkg")
    rc, out = sh(["camber","stash","cp",str(tar),f"stash://{a.stash_path}/pkg.tar.gz"])
    if rc != 0: sh(["camber","stash","upload",str(tar),f"{a.stash_path}/"], check=True)
    jid, out = submit("tar xzf pkg.tar.gz && cd pkg && bash gpu/run.sh", a.engine, a.size, a.stash_path, gpu=True)
    if not jid: raise SystemExit("could not parse job id")
    st = wait(jid); sh(["camber","job","log",jid,str(outdir)])
    # outputs: pkg/out.tar.gz written by run.sh into the stash
    rc, out = sh(["camber","stash","cp",f"stash://{a.stash_path}/pkg/out.tar.gz",str(outdir/"out.tar.gz")])
    if rc != 0: sh(["camber","stash","download",f"{a.stash_path}/pkg/out.tar.gz",str(outdir)])
    o = outdir/"out.tar.gz"; n = 0
    if o.exists():
        with tarfile.open(o) as t: t.extractall(outdir/"out")
        for f in (outdir/"out"/"generated").rglob("*"):
            if f.is_file(): d = gen/f.relative_to(outdir/"out"/"generated"); d.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, d); n += 1
        stp = proj/"status.json"; status = json.loads(stp.read_text()) if stp.exists() else {}
        ks = outdir/"out"/"gpu_status.json"
        if ks.exists():
            for k, v in json.loads(ks.read_text()).items(): status[k] = {**v, "worker": "camber"}
            stp.write_text(json.dumps(status, indent=2))
        shutil.rmtree(outdir/"out", ignore_errors=True); o.unlink()
    shutil.rmtree(pkg, ignore_errors=True); tar.unlink(missing_ok=True)
    print(f"job {jid} -> {st}; merged {n} files"); return 0 if n else 1

if __name__ == "__main__": sys.exit(main())

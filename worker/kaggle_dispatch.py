#!/usr/bin/env python3
"""
Runs on GitHub Actions. Pushes worker/kaggle/ as a Kaggle *script kernel* (GPU + internet), waits, downloads outputs,
merges them into projects/<project>/generated/ and status.json. $0 — Kaggle free GPU quota (30 h/week).
  python3 worker/kaggle_dispatch.py --project ep1_hinterkaifeck --stage image [--only a,b] [--gpu T4x2|P100]
Env: KAGGLE_USERNAME, KAGGLE_KEY (free account, phone-verified for GPU).
"""
import os, sys, json, time, shutil, pathlib, argparse, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]
KDIR = ROOT / "worker" / "kaggle"

def sh(cmd, check=True, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if check and r.returncode != 0: raise SystemExit(f"$ {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return (r.stdout + r.stderr).strip()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--project", required=True); ap.add_argument("--stage", default="image")
    ap.add_argument("--only", default=""); ap.add_argument("--gpu", default="T4x2"); ap.add_argument("--timeout-min", type=int, default=600)
    a = ap.parse_args()
    user = os.environ.get("KAGGLE_USERNAME"); key = os.environ.get("KAGGLE_KEY")
    if not (user and key): raise SystemExit("KAGGLE_USERNAME / KAGGLE_KEY not set (GitHub Secrets)")
    repo_url = sh(["git", "remote", "get-url", "origin"], cwd=ROOT).replace("git@github.com:", "https://github.com/").removesuffix(".git")
    slug = f"room313-{a.project}-{a.stage}".lower().replace("_", "-")[:50]
    # kernel payload
    job = json.dumps({"repo": repo_url, "project": a.project, "stage": a.stage, "only": [x for x in a.only.split(",") if x]})
    push = ROOT / "quality_reports" / "_kaggle_push"; shutil.rmtree(push, ignore_errors=True); push.mkdir(parents=True)
    (push / "script.py").write_text((KDIR / "room313_worker.py").read_text().replace("__JOB_JSON__", job.replace("\\", "\\\\").replace('"', '\\"')).replace("__HF_TOKEN__", os.environ.get("HF_TOKEN", "")))
    (push / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug, "code_file": "script.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true", "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=1))
    print(sh(["kaggle", "kernels", "push", "-p", str(push)])); shutil.rmtree(push, ignore_errors=True)
    kid = f"{user}/{slug}"; t0 = time.time(); last = ""
    while True:
        st = sh(["kaggle", "kernels", "status", kid], check=False)
        if st != last: print(f"  {int(time.time()-t0)}s {st}", flush=True); last = st
        s = st.lower()
        if "complete" in s or "error" in s or "cancel" in s: break
        if time.time() - t0 > a.timeout_min * 60: print("TIMEOUT waiting for kernel"); break
        time.sleep(30)
    outdir = ROOT / "quality_reports" / f"kaggle_{slug}_{int(t0)}"; outdir.mkdir(parents=True, exist_ok=True)
    print(sh(["kaggle", "kernels", "output", kid, "-p", str(outdir)], check=False))
    # merge outputs
    gen = ROOT / "projects" / a.project / "generated"; n = 0
    for f in (outdir / "generated").rglob("*"):
        if f.is_file(): dst = gen / f.relative_to(outdir / "generated"); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, dst); n += 1
    stp = ROOT / "projects" / a.project / "status.json"; status = json.loads(stp.read_text()) if stp.exists() else {}
    ks = outdir / "kaggle_status.json"
    if ks.exists():
        for k, v in json.loads(ks.read_text()).items(): status[k] = {**v, "worker": "kaggle"}
        stp.write_text(json.dumps(status, indent=2))
    for p in (outdir / "generated",): shutil.rmtree(p, ignore_errors=True)   # assets live in projects/, keep only logs/status here
    print(f"merged {n} files into {gen}; kernel log: {outdir}")
    bad = [k for k, v in status.items() if v.get("worker") == "kaggle" and v.get("state") != "ok"]
    print("not ok:", bad)
    return 0 if n and not bad else 1

if __name__ == "__main__": sys.exit(main())

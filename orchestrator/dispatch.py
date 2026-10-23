#!/usr/bin/env python3
"""
Trigger a GitHub Actions workflow, wait for it, pull logs/artifacts/assets back.
  python3 orchestrator/dispatch.py connectivity
  python3 orchestrator/dispatch.py free-audit [--no-cpu-test] [--no-space-tests]
  python3 orchestrator/dispatch.py generate <project> <stage> [--dry-run false] [--only id1,id2]
      stage: milestone | all | character | image | video | audio
Requires: gh CLI authenticated for this repo and the repo as `origin`. Everything is $0 — there is no budget/authorize flag any more.
"""
import sys, json, time, subprocess, argparse, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
WF = {"connectivity": "connectivity.yml", "free-audit": "free-audit.yml", "generate": "generate.yml"}

def sh(cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if check and r.returncode != 0: raise SystemExit(f"$ {' '.join(cmd)}\n{r.stderr}")
    return r.stdout.strip()

def repo():
    url = sh(["git","remote","get-url","origin"])
    return url.replace("https://github.com/","").replace("git@github.com:","").removesuffix(".git")

# REST only (gh's `workflow run` / `run list` need GraphQL, which some proxies block)
def api(path, *args): return json.loads(sh(["gh","api",path,*args]) or "null")
def run_ids(workflow): return {r["id"] for r in api(f"repos/{repo()}/actions/workflows/{workflow}/runs?per_page=20").get("workflow_runs", [])}
def trigger(workflow, inputs):
    args = ["-f", "ref=main"] + [x for k, v in inputs.items() for x in ("-f", f"inputs[{k}]={v}")]
    sh(["gh","api","-X","POST",f"repos/{repo()}/actions/workflows/{workflow}/dispatches",*args])

def latest_run(workflow, before_ids):
    for _ in range(30):
        new = sorted(run_ids(workflow) - before_ids)
        if new: return new[-1]
        time.sleep(4)
    raise SystemExit("run did not appear")

def wait(run_id, poll=20):
    while True:
        r = api(f"repos/{repo()}/actions/runs/{run_id}")
        print(f"  run {run_id}: {r['status']} {r.get('conclusion') or ''}", flush=True)
        if r["status"] == "completed": return r["conclusion"]
        time.sleep(poll)

def job_logs(run_id):
    out = []
    for j in api(f"repos/{repo()}/actions/runs/{run_id}/jobs").get("jobs", []):
        out.append(f"### job {j['name']} {j['conclusion']}")
        for st in j.get("steps", []): out.append(f"  {st['conclusion'] or st['status']:8s} {st['name']}")
    return "\n".join(out)

def download_artifacts(run_id, dest):
    import io, zipfile
    for a in api(f"repos/{repo()}/actions/runs/{run_id}/artifacts").get("artifacts", []):
        r = subprocess.run(["gh","api",f"repos/{repo()}/actions/artifacts/{a['id']}/zip"], capture_output=True, cwd=ROOT)
        if r.returncode == 0 and r.stdout[:2] == b"PK":
            d = dest/a["name"]; d.mkdir(parents=True, exist_ok=True); zipfile.ZipFile(io.BytesIO(r.stdout)).extractall(d); print("artifact", a["name"], "->", d)
        else: print("artifact", a["name"], "download failed:", r.stderr.decode()[:200])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=WF); ap.add_argument("project", nargs="?"); ap.add_argument("stage", nargs="?", default="milestone")
    ap.add_argument("--dry-run", default="true"); ap.add_argument("--only", default="")
    ap.add_argument("--no-cpu-test", action="store_true"); ap.add_argument("--no-space-tests", action="store_true")
    a = ap.parse_args(); wf = WF[a.cmd]
    before = run_ids(wf)
    inputs = {}
    if a.cmd == "free-audit":
        inputs = {"run_cpu_image_test": "false" if a.no_cpu_test else "true", "run_space_tests": "false" if a.no_space_tests else "true"}
    elif a.cmd == "generate":
        if not a.project: raise SystemExit("generate needs <project>")
        inputs = {"project": a.project, "stage": a.stage, "dry_run": a.dry_run, "only": a.only}
    trigger(wf, inputs)
    rid = latest_run(wf, before); print("dispatched run", rid)
    concl = wait(rid)
    qr = ROOT/"quality_reports"; qr.mkdir(exist_ok=True)
    print(job_logs(rid))
    dest = qr/f"{a.cmd}_{rid}"; download_artifacts(rid, dest)
    if a.cmd == "generate":
        print(sh(["git","pull","--rebase","origin","main"], check=False))
        st = ROOT/"projects"/a.project/"status.json"
        if st.exists():
            s = json.loads(st.read_text()); bad = {k:v for k,v in s.items() if v.get("state") != "ok"}
            print(f"status: {len(s)-len(bad)} ok, {len(bad)} not ok", json.dumps(bad, ensure_ascii=False, indent=1)[:2000])
    print("conclusion:", concl); return 0 if concl == "success" else 1

if __name__ == "__main__": sys.exit(main())

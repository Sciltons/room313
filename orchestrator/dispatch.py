#!/usr/bin/env python3
"""
Trigger the GitHub Actions worker, wait for it, pull the generated assets back.
  python3 orchestrator/dispatch.py connectivity
  python3 orchestrator/dispatch.py generate <project> <stage> --budget 1.0 [--authorize YES] [--dry-run false] [--only id1,id2]
Requires: gh CLI authenticated for this repo (repo-scoped token is enough) and the repo as `origin`.
"""
import sys, json, time, subprocess, argparse, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]

def sh(cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if check and r.returncode != 0: raise SystemExit(f"$ {' '.join(cmd)}\n{r.stderr}")
    return r.stdout.strip()

def repo():
    url = sh(["git","remote","get-url","origin"]);
    return url.replace("https://github.com/","").replace("git@github.com:","").removesuffix(".git")

def latest_run(workflow, before_ids):
    for _ in range(30):
        runs = json.loads(sh(["gh","run","list","-R",repo(),"--workflow",workflow,"--limit","5","--json","databaseId,status,conclusion,createdAt"]))
        new = [r for r in runs if r["databaseId"] not in before_ids]
        if new: return new[0]["databaseId"]
        time.sleep(4)
    raise SystemExit("run did not appear")

def wait(run_id, poll=20):
    while True:
        r = json.loads(sh(["gh","run","view","-R",repo(),str(run_id),"--json","status,conclusion"]))
        print(f"  run {run_id}: {r['status']} {r.get('conclusion') or ''}", flush=True)
        if r["status"] == "completed": return r["conclusion"]
        time.sleep(poll)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("project", nargs="?"); ap.add_argument("stage", nargs="?", default="all")
    ap.add_argument("--budget", default="0"); ap.add_argument("--authorize", default=""); ap.add_argument("--dry-run", default="true"); ap.add_argument("--only", default="")
    a = ap.parse_args()
    wf = "connectivity.yml" if a.cmd == "connectivity" else "generate.yml"
    before = {r["databaseId"] for r in json.loads(sh(["gh","run","list","-R",repo(),"--workflow",wf,"--limit","20","--json","databaseId"]) or "[]")}
    if a.cmd == "connectivity":
        sh(["gh","workflow","run",wf,"-R",repo()])
    else:
        if a.dry_run.lower() == "false" and a.authorize != "YES": raise SystemExit("paid run needs --authorize YES")
        sh(["gh","workflow","run",wf,"-R",repo(),"-f",f"project={a.project}","-f",f"stage={a.stage}","-f",f"dry_run={a.dry_run}","-f",f"authorize={a.authorize}","-f",f"budget_usd={a.budget}","-f",f"only={a.only}"])
    rid = latest_run(wf, before); print("dispatched run", rid)
    concl = wait(rid)
    log = sh(["gh","run","view","-R",repo(),str(rid),"--log"], check=False)
    (ROOT/"quality_reports").mkdir(exist_ok=True); (ROOT/"quality_reports"/f"run_{rid}.log").write_text(log)
    tail = [l.split("\t")[-1] for l in log.splitlines() if "\t" in l][-60:]
    print("\n".join(tail))
    if a.cmd != "connectivity":
        print(sh(["git","pull","--rebase","origin","main"], check=False))
    print("conclusion:", concl); return 0 if concl == "success" else 1

if __name__ == "__main__": sys.exit(main())

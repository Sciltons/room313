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

def latest_run(workflow, before_ids):
    for _ in range(30):
        runs = json.loads(sh(["gh","run","list","-R",repo(),"--workflow",workflow,"--limit","5","--json","databaseId,status,conclusion,createdAt"]) or "[]")
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
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=WF); ap.add_argument("project", nargs="?"); ap.add_argument("stage", nargs="?", default="milestone")
    ap.add_argument("--dry-run", default="true"); ap.add_argument("--only", default="")
    ap.add_argument("--no-cpu-test", action="store_true"); ap.add_argument("--no-space-tests", action="store_true")
    a = ap.parse_args(); wf = WF[a.cmd]
    before = {r["databaseId"] for r in json.loads(sh(["gh","run","list","-R",repo(),"--workflow",wf,"--limit","20","--json","databaseId"]) or "[]")}
    cmd = ["gh","workflow","run",wf,"-R",repo()]
    if a.cmd == "free-audit":
        cmd += ["-f", f"run_cpu_image_test={'false' if a.no_cpu_test else 'true'}", "-f", f"run_space_tests={'false' if a.no_space_tests else 'true'}"]
    elif a.cmd == "generate":
        if not a.project: raise SystemExit("generate needs <project>")
        cmd += ["-f",f"project={a.project}","-f",f"stage={a.stage}","-f",f"dry_run={a.dry_run}","-f",f"only={a.only}"]
    sh(cmd)
    rid = latest_run(wf, before); print("dispatched run", rid)
    concl = wait(rid)
    qr = ROOT/"quality_reports"; qr.mkdir(exist_ok=True)
    log = sh(["gh","run","view","-R",repo(),str(rid),"--log"], check=False)
    (qr/f"run_{rid}.log").write_text(log)
    tail = [l.split("\t")[-1] for l in log.splitlines() if "\t" in l][-60:]
    print("\n".join(tail))
    # artifacts (audit reports, probe.csv, run.log)
    dest = qr/f"{a.cmd}_{rid}"; dest.mkdir(exist_ok=True)
    print(sh(["gh","run","download","-R",repo(),str(rid),"-D",str(dest)], check=False) or f"artifacts -> {dest}")
    if a.cmd == "generate":
        print(sh(["git","pull","--rebase","origin","main"], check=False))
        st = ROOT/"projects"/a.project/"status.json"
        if st.exists():
            s = json.loads(st.read_text()); bad = {k:v for k,v in s.items() if v.get("state") != "ok"}
            print(f"status: {len(s)-len(bad)} ok, {len(bad)} not ok", json.dumps(bad, ensure_ascii=False, indent=1)[:2000])
    print("conclusion:", concl); return 0 if concl == "success" else 1

if __name__ == "__main__": sys.exit(main())

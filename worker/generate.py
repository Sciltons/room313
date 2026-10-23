#!/usr/bin/env python3
"""
External generation worker (runs on GitHub Actions).

Reads  projects/<project>/manifest.json  (list of generation jobs)
Writes projects/<project>/generated/...   (assets)  +  ledger.json / status.json

Safety:
  * dry_run=true  -> validates manifest, prints plan + cost estimate, spends nothing.
  * authorize must equal "YES" for any paid call.
  * hard budget cap: stops before a job that would exceed `budget_usd` (cumulative ledger).
  * resume: a job whose output file already exists is skipped.
  * retries: `max_retries` per job with backoff (3 min+ on Space quota errors); failures recorded, never fabricated.
  * stage selection: `all` | job type | job stage | any tag (e.g. `milestone` = 1 character + 1 keyframe + 1 shot).
Providers (ZERO-COST mode, default): hf_space (public Hugging Face Spaces via gradio_client, free quota),
  local_cpu (open-source diffusers / audiocraft models on the runner CPU), kaggle (free GPU kernel via the official Kaggle API),
  freesound (CC0/CC-BY library). Paid providers (replicate, elevenlabs_sfx, fal) are hard-disabled unless ALLOW_PAID=1 — never set.
"""
import os, sys, json, time, base64, hashlib, argparse, mimetypes, subprocess, pathlib
import requests, shutil
ZERO_COST = os.environ.get("ALLOW_PAID","0") != "1"
PAID = {"replicate","elevenlabs_sfx","fal","runway","kling"}

ROOT = pathlib.Path(__file__).resolve().parents[1]

def log(*a): print(*a, flush=True)

# ---------------------------------------------------------------- cost table (USD, conservative estimates)
COST = {
    "black-forest-labs/flux-schnell": 0.003,
    "black-forest-labs/flux-dev": 0.025,
    "black-forest-labs/flux-1.1-pro": 0.04,
    "black-forest-labs/flux-kontext-pro": 0.04,
    "black-forest-labs/flux-kontext-max": 0.08,
    "kwaivgi/kling-v2.1": 0.55,            # per 5 s standard
    "kwaivgi/kling-v1.6-standard": 0.30,
    "wan-video/wan-2.2-i2v-fast": 0.10,
    "minimax/video-01": 0.50,
    "stackadoc/stable-audio-open-1.0": 0.02,
    "meta/musicgen": 0.08,
    "elevenlabs_sfx": 0.02,
}

class Budget:
    def __init__(self, ledger_path, cap):
        self.path = ledger_path; self.cap = cap
        self.ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"spent_usd": 0.0, "entries": []}
    @property
    def spent(self): return self.ledger["spent_usd"]
    def can(self, est): return self.spent + est <= self.cap + 1e-9
    def charge(self, job_id, est, meta):
        self.ledger["spent_usd"] = round(self.spent + est, 4)
        self.ledger["entries"].append({"job": job_id, "usd": est, "ts": time.time(), **meta})
        self.path.write_text(json.dumps(self.ledger, indent=2))

# ---------------------------------------------------------------- providers
def file_to_data_url(p):
    mt = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
    return f"data:{mt};base64," + base64.b64encode(pathlib.Path(p).read_bytes()).decode()

def resolve_inputs(inp, gen_dir, as_path=False):
    """replace 'asset:<relpath>' strings (at any depth) with data URLs — or absolute paths for hf_space (gradio handle_file)"""
    def conv(v):
        if isinstance(v, str) and v.startswith("asset:"):
            p = gen_dir / v[6:]
            if not p.exists(): raise FileNotFoundError(f"dependency missing: {v}")
            return str(p.resolve()) if as_path else file_to_data_url(p)
        if isinstance(v, dict): return {k: conv(x) for k, x in v.items()}
        if isinstance(v, list): return [conv(x) for x in v]
        return v
    return conv(inp)

def replicate_run(model, inp, token, timeout=1800):
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Prefer": "wait=60"}
    if ":" in model:   # pinned version hash
        r = requests.post("https://api.replicate.com/v1/predictions", headers=h, json={"version": model.split(":")[1], "input": inp}, timeout=120)
    else:
        r = requests.post(f"https://api.replicate.com/v1/models/{model}/predictions", headers=h, json={"input": inp}, timeout=120)
    if r.status_code >= 400: raise RuntimeError(f"replicate {r.status_code}: {r.text[:400]}")
    pred = r.json(); url = pred["urls"]["get"]; t0 = time.time()
    while pred["status"] not in ("succeeded", "failed", "canceled"):
        if time.time() - t0 > timeout: raise TimeoutError("replicate prediction timeout")
        time.sleep(5); pred = requests.get(url, headers=h, timeout=60).json()
    if pred["status"] != "succeeded": raise RuntimeError(f"replicate failed: {pred.get('error')}")
    out = pred["output"]
    if isinstance(out, list): out = out[0]
    if isinstance(out, dict) and "url" in out: out = out["url"]
    return out, pred.get("metrics", {})

def elevenlabs_sfx(inp, key):
    r = requests.post("https://api.elevenlabs.io/v1/sound-generation", headers={"xi-api-key": key},
                      json={"text": inp["prompt"], "duration_seconds": inp.get("duration", 5), "prompt_influence": inp.get("prompt_influence", 0.4)}, timeout=300)
    if r.status_code >= 400: raise RuntimeError(f"elevenlabs {r.status_code}: {r.text[:300]}")
    return r.content

QUOTA_WORDS = ("quota", "gpu", "rate limit", "too many", "429", "queue is full", "exceeded")
def is_quota_error(e): return any(w in str(e).lower() for w in QUOTA_WORDS)

def hf_space_run(inp, token, out):
    """inp: {space, api_name, args:{...}, result_index, api_hint}. Returns local file path produced by the Space.
    Tolerant: if api_name is not exposed, picks the endpoint whose name contains api_hint (or the first one);
    args not accepted by the endpoint are dropped (reported in metrics) instead of crashing."""
    from gradio_client import Client, handle_file
    try: c = Client(inp["space"], token=token or None, verbose=False)
    except TypeError: c = Client(inp["space"], hf_token=token or None, verbose=False)   # older gradio_client
    api = c.view_api(return_format="dict", print_info=False) or {}
    named = api.get("named_endpoints", {})
    api_name = inp["api_name"]; note = {}
    if named and api_name not in named:
        hint = inp.get("api_hint", "").lower()
        cands = [n for n in named if hint and hint in n.lower()] or list(named)
        note["api_name_fallback"] = f"{api_name} -> {cands[0]}"; api_name = cands[0]
    accepted = {p.get("parameter_name") for p in named.get(api_name, {}).get("parameters", [])} if named else None
    args = {}; dropped = []
    for k, v in inp.get("args", {}).items():
        if accepted is not None and k not in accepted: dropped.append(k); continue
        args[k] = handle_file(v) if isinstance(v, str) and (v.startswith("/") or v.startswith("data:") or v.startswith("http")) else v
    if dropped: note["dropped_args"] = dropped
    r = c.predict(**args, api_name=api_name)
    idx = inp.get("result_index", 0)
    if isinstance(r, (list, tuple)): r = r[idx]
    if isinstance(r, dict): r = r.get("video") or r.get("path") or r.get("url") or r
    if isinstance(r, str) and r.startswith("http"): download(r, out)
    else: out.parent.mkdir(parents=True, exist_ok=True); shutil.copy(r, out)
    return {"space": inp["space"], "api_name": api_name, **note}

def local_cpu_run(inp, out):
    """open-source models on the runner CPU. kind: lcm_image | musicgen"""
    import torch, time
    t0 = time.time(); out.parent.mkdir(parents=True, exist_ok=True)
    if inp["kind"] == "lcm_image":
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(inp.get("model_id", "SimianLuo/LCM_Dreamshaper_v7"), torch_dtype=torch.float32).to("cpu")
        im = pipe(prompt=inp["prompt"], num_inference_steps=inp.get("steps", 6), guidance_scale=inp.get("cfg", 1.5), width=inp.get("width", 768), height=inp.get("height", 432)).images[0]
        im.save(out)
    elif inp["kind"] == "musicgen":   # facebook/musicgen-small (CC-BY-NC weights: non-commercial use only — flagged in metadata)
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        import scipy.io.wavfile as w
        proc = AutoProcessor.from_pretrained("facebook/musicgen-small"); m = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        x = proc(text=[inp["prompt"]], padding=True, return_tensors="pt"); a = m.generate(**x, max_new_tokens=int(inp.get("duration", 10) * 50))
        w.write(out, m.config.audio_encoder.sampling_rate, a[0, 0].numpy())
    else: raise RuntimeError("unknown local kind")
    return {"elapsed": round(time.time() - t0, 1), "license_note": "CC-BY-NC" if inp["kind"] == "musicgen" else "model license"}

def freesound_run(inp, key, out):
    """search CC0 sounds and download the best preview (HQ mp3). Attribution recorded in metadata."""
    q = {"query": inp["query"], "filter": inp.get("filter", 'license:"Creative Commons 0"'), "fields": "id,name,duration,license,username,previews", "sort": "rating_desc", "token": key}
    r = requests.get("https://freesound.org/apiv2/search/text/", params=q, timeout=60); r.raise_for_status()
    res = [x for x in r.json()["results"] if inp.get("min_dur", 0) <= x["duration"] <= inp.get("max_dur", 600)]
    if not res: raise RuntimeError("no freesound result")
    best = res[0]; download(best["previews"]["preview-hq-mp3"], out)
    return {"freesound_id": best["id"], "name": best["name"], "license": best["license"], "by": best["username"]}

def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16): f.write(chunk)

# ---------------------------------------------------------------- QC (mechanical checks on the runner)
def qc(job, path):
    if not path.exists() or path.stat().st_size < 1024: return False, "empty output"
    if job["type"] == "image":
        try:
            from PIL import Image
            im = Image.open(path); w, h = im.size
            if w < 512 or h < 512: return False, f"too small {w}x{h}"
        except Exception as e: return False, f"unreadable image: {e}"
    elif job["type"] in ("video", "audio"):
        p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
        try: d = float(p.stdout.strip())
        except: return False, "ffprobe failed"
        want = job.get("expect_duration")
        if want and d < want * 0.8: return False, f"duration {d:.1f}s < expected {want}s"
        if job["type"] == "video":
            b = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-vf", "blackdetect=d=0.5:pix_th=0.05", "-an", "-f", "null", "-"], capture_output=True, text=True).stderr
            if "black_start:0" in b.replace(" ", ""): return False, "starts with black frames"
    return True, "ok"

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True); ap.add_argument("--stage", default="all")
    ap.add_argument("--budget-usd", type=float, default=0.0); ap.add_argument("--dry-run", default="true")
    ap.add_argument("--authorize", default=""); ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--only", default="")   # comma-separated job ids
    a = ap.parse_args()
    dry = a.dry_run.lower() == "true"
    proj = ROOT / "projects" / a.project; gen = proj / "generated"; gen.mkdir(parents=True, exist_ok=True)
    man = json.loads((proj / "manifest.json").read_text())
    budget = Budget(proj / "ledger.json", a.budget_usd)
    status = json.loads((proj / "status.json").read_text()) if (proj / "status.json").exists() else {}
    tokens = {"replicate": os.environ.get("REPLICATE_API_TOKEN", ""), "elevenlabs": os.environ.get("ELEVENLABS_API_KEY", ""),
              "hf": os.environ.get("HF_TOKEN", ""), "freesound": os.environ.get("FREESOUND_API_KEY", "")}
    if ZERO_COST:
        paid = [j["id"] for j in man["jobs"] if j.get("provider", "replicate") in PAID]
        if paid: log(f"ZERO-COST MODE: refusing {len(paid)} paid jobs: {paid[:5]}..."); 
        man["jobs"] = [j for j in man["jobs"] if j.get("provider", "replicate") not in PAID]
        a.budget_usd = 0.0
    only = set(x for x in a.only.split(",") if x)

    def in_stage(j): return a.stage == "all" or j["type"] == a.stage or j.get("stage") == a.stage or a.stage in j.get("tags", [])
    jobs = [j for j in man["jobs"] if in_stage(j) and (not only or j["id"] in only)]
    if not jobs: log(f"no jobs match stage={a.stage!r} only={a.only!r}"); return 1
    pending = [j for j in jobs if not (gen / j["out"]).exists()]
    est_total = sum(COST.get(j["model"], j.get("est_usd", 0.1)) * j.get("units", 1) for j in pending)
    log(f"project={a.project} jobs={len(jobs)} done={len(jobs)-len(pending)} pending={len(pending)} est=${est_total:.2f} spent=${budget.spent:.2f} cap=${a.budget_usd:.2f} dry_run={dry}")
    for j in pending: log(f"  - {j['id']:28s} {j['type']:6s} {j['model']:40s} ~${COST.get(j['model'], j.get('est_usd',0.1))*j.get('units',1):.3f} -> {j['out']}")
    if dry:
        log("DRY RUN: nothing generated, nothing spent."); return 0
    if not ZERO_COST and a.authorize != "YES":
        log("REFUSED: paid generation requires authorize=YES"); return 2

    failures = 0
    for j in pending:
        est = 0.0 if j.get("provider") in ("hf_space","local_cpu","kaggle","freesound") else COST.get(j["model"], j.get("est_usd", 0.1)) * j.get("units", 1)
        if not budget.can(est):
            log(f"BUDGET STOP before {j['id']}: spent ${budget.spent:.2f} + ${est:.2f} > cap ${a.budget_usd:.2f}"); status[j["id"]] = {"state": "budget_hold"}; break
        out = gen / j["out"]
        for attempt in range(1, a.max_retries + 2):
            try:
                log(f"[{j['id']}] attempt {attempt} model={j['model']}")
                prov = j.get("provider", "replicate")
                inp = resolve_inputs(j["input"], gen, as_path=(prov == "hf_space"))
                if prov == "hf_space": metrics = hf_space_run(inp, tokens["hf"], out)
                elif prov == "local_cpu": metrics = local_cpu_run(inp, out)
                elif prov == "freesound": metrics = freesound_run(inp, tokens["freesound"], out)
                elif prov == "replicate":
                    url, metrics = replicate_run(j["model"], inp, tokens["replicate"])
                    download(url, out)
                elif prov == "elevenlabs_sfx":
                    if not tokens["elevenlabs"]: raise RuntimeError("ELEVENLABS_API_KEY missing")
                    out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(elevenlabs_sfx(inp, tokens["elevenlabs"])); metrics = {}
                else: raise RuntimeError(f"unknown provider {prov}")
                budget.charge(j["id"], est, {"model": j["model"], "attempt": attempt, "metrics": metrics})
                ok, why = qc(j, out)
                status[j["id"]] = {"state": "ok" if ok else "qc_failed", "qc": why, "attempt": attempt, "out": j["out"]}
                log(f"[{j['id']}] {'OK' if ok else 'QC FAILED: '+why}")
                if not ok and attempt <= a.max_retries: out.unlink(missing_ok=True); continue
                break
            except Exception as e:
                quota = is_quota_error(e)
                log(f"[{j['id']}] ERROR{' (quota/GPU limit)' if quota else ''}: {str(e)[:500]}")
                status[j["id"]] = {"state": "quota" if quota else "error", "error": str(e)[:300], "attempt": attempt}
                if attempt <= a.max_retries: time.sleep((180 if quota else 10) * attempt); continue
                failures += 1
        (proj / "status.json").write_text(json.dumps(status, indent=2))
    (proj / "status.json").write_text(json.dumps(status, indent=2))
    log(f"done. failures={failures} spent=${budget.spent:.2f}")
    return 0 if failures == 0 else 1

if __name__ == "__main__": sys.exit(main())

#!/usr/bin/env python3
"""
Room 313 — Kaggle GPU worker (runs INSIDE a Kaggle kernel, free T4/P100).
Reads job.json (written by worker/kaggle_dispatch.py before the push):
  {"repo": "https://github.com/<owner>/<repo>", "project": "...", "stage": "image|video|character|milestone|all", "only": [..]}
Clones the public repo, reads projects/<project>/manifest.json, runs the matching jobs locally on the GPU
(FLUX.1-schnell for images, LTX-Video for image->video), writes outputs to /kaggle/working/generated/<out>
plus /kaggle/working/kaggle_status.json. Everything is $0 (Kaggle free GPU quota).
"""
import os, sys, json, time, subprocess, pathlib, shutil, traceback

WORK = pathlib.Path("/kaggle/working"); GEN = WORK / "generated"; GEN.mkdir(parents=True, exist_ok=True)
JOB = json.loads(os.environ.get("ROOM313_JOB") or "__JOB_JSON__")   # injected by kaggle_dispatch.py (kernels carry a single file)
STATUS = {}; T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
def save_status(): (WORK / "kaggle_status.json").write_text(json.dumps(STATUS, indent=1))

# ---------------------------------------------------------------- repo + manifest
REPO = WORK / "repo"
if REPO.exists(): shutil.rmtree(REPO)
subprocess.run(["git", "clone", "--depth", "1", JOB["repo"], str(REPO)], check=True)
proj = REPO / "projects" / JOB["project"]; man = json.loads((proj / "manifest.json").read_text())
prev_gen = proj / "generated"
stage, only = JOB.get("stage", "all"), set(JOB.get("only", []))
def in_stage(j): return stage == "all" or j["type"] == stage or j.get("stage") == stage or stage in j.get("tags", [])
jobs = [j for j in man["jobs"] if in_stage(j) and (not only or j["id"] in only) and j.get("provider") == "hf_space"]
pending = [j for j in jobs if not (prev_gen / j["out"]).exists() or j["id"] in only]
log(f"stage={stage} jobs={len(jobs)} pending={len(pending)} gpu={subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()}")
if not pending: save_status(); sys.exit(0)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers>=0.32", "transformers>=4.46", "accelerate", "bitsandbytes>=0.44", "sentencepiece", "protobuf", "imageio[ffmpeg]", "pillow"], check=False)
import torch
from PIL import Image
DEV = "cuda"

# ---------------------------------------------------------------- FLUX.1-schnell (Apache-2.0), nf4 on T4 (16 GB)
_flux = None
def flux():
    global _flux
    if _flux is None:
        from diffusers import FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig as DBnB
        from transformers import T5EncoderModel, BitsAndBytesConfig as TBnB
        mid = "black-forest-labs/FLUX.1-schnell"; dt = torch.float16
        log("loading FLUX.1-schnell nf4 ...")
        t5 = T5EncoderModel.from_pretrained(mid, subfolder="text_encoder_2", quantization_config=TBnB(load_in_8bit=True), torch_dtype=dt)
        tr = FluxTransformer2DModel.from_pretrained(mid, subfolder="transformer", quantization_config=DBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt), torch_dtype=dt)
        _flux = FluxPipeline.from_pretrained(mid, transformer=tr, text_encoder_2=t5, torch_dtype=dt)
        _flux.enable_model_cpu_offload(); _flux.vae.enable_slicing(); _flux.vae.enable_tiling()
        log("FLUX ready")
    return _flux

def gen_image(j, out):
    a = j["input"]["args"]; p = flux()
    g = torch.Generator("cpu").manual_seed(int(a.get("seed", 0)))
    im = p(prompt=a["prompt"], width=int(a.get("width", 1024)), height=int(a.get("height", 576)), num_inference_steps=int(a.get("num_inference_steps", 4)), guidance_scale=0.0, generator=g, max_sequence_length=256).images[0]
    if im.getextrema() == ((0, 0), (0, 0), (0, 0)): raise RuntimeError("black image (fp16 overflow?)")
    out.parent.mkdir(parents=True, exist_ok=True); im.save(out); return {"model": "FLUX.1-schnell nf4", "seed": a.get("seed")}

# ---------------------------------------------------------------- LTX-Video image->video (Lightricks, 2B, fp16)
_ltx = None
def ltx():
    global _ltx
    if _ltx is None:
        from diffusers import LTXImageToVideoPipeline
        log("loading LTX-Video ...")
        _ltx = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video-0.9.7-distilled", torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        _ltx.enable_model_cpu_offload(); _ltx.vae.enable_tiling()
        log("LTX ready")
    return _ltx

def gen_video(j, out):
    from diffusers.utils import export_to_video
    a = j["input"]["args"]; p = ltx()
    img_rel = a["input_image_filepath"].replace("asset:", "")
    src = GEN / img_rel if (GEN / img_rel).exists() else prev_gen / img_rel
    if not src.exists(): raise FileNotFoundError(f"keyframe missing: {img_rel}")
    W, H = int(a.get("width_ui", 896)), int(a.get("height_ui", 512)); W -= W % 32; H -= H % 32
    img = Image.open(src).convert("RGB").resize((W, H))
    secs = float(a.get("duration_ui", 2)); fps = 24; frames = int(secs * fps); frames = (frames // 8) * 8 + 1
    g = torch.Generator("cpu").manual_seed(int(a.get("seed_ui", 42)))
    vid = p(image=img, prompt=a["prompt"], negative_prompt=a.get("negative_prompt", ""), width=W, height=H, num_frames=frames, num_inference_steps=8, guidance_scale=1.0, generator=g).frames[0]
    out.parent.mkdir(parents=True, exist_ok=True); export_to_video(vid, str(out), fps=fps); return {"model": "LTX-Video-0.9.7-distilled", "frames": frames, "fps": fps}

# ---------------------------------------------------------------- run
fails = 0
for j in pending:
    out = GEN / j["out"]
    try:
        log(f"[{j['id']}] {j['type']} ...")
        m = gen_image(j, out) if j["type"] == "image" else gen_video(j, out) if j["type"] == "video" else None
        if m is None: STATUS[j["id"]] = {"state": "skipped", "why": "unsupported type on kaggle"}; continue
        STATUS[j["id"]] = {"state": "ok", "out": j["out"], "metrics": m, "elapsed": round(time.time() - T0, 1)}; log(f"[{j['id']}] OK")
    except Exception as e:
        fails += 1; STATUS[j["id"]] = {"state": "error", "error": str(e)[:400]}; log(f"[{j['id']}] ERROR {e}"); traceback.print_exc()
    save_status()
    torch.cuda.empty_cache()
shutil.rmtree(REPO, ignore_errors=True)
log(f"done. ok={sum(1 for v in STATUS.values() if v['state']=='ok')} failed={fails}")

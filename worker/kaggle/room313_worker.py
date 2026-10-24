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
# No HF token in the kernel source: the official FLUX repo is gated; we use an unmodified Apache-2.0 mirror instead.
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
def in_stage(j): return stage == "all" or j["type"] == stage or j.get("stage") == stage or stage in j.get("tags", []) or (stage == "images" and j["type"] == "image")
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
        dt = torch.float16
        # official repo first; if it is gated for this account, fall back to an unmodified Apache-2.0 mirror in diffusers format
        for mid in [os.environ.get("FLUX_REPO") or "YuCollection/FLUX.1-schnell-Diffusers", "black-forest-labs/FLUX.1-schnell"]:
            try:
                log(f"loading {mid} nf4 ...")
                t5 = T5EncoderModel.from_pretrained(mid, subfolder="text_encoder_2", quantization_config=TBnB(load_in_8bit=True), torch_dtype=dt)
                tr = FluxTransformer2DModel.from_pretrained(mid, subfolder="transformer", quantization_config=DBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt), torch_dtype=dt)
                _flux = FluxPipeline.from_pretrained(mid, transformer=tr, text_encoder_2=t5, torch_dtype=dt); break
            except Exception as e:
                if "gated" in str(e).lower() or "403" in str(e) or "401" in str(e): log(f"  {mid}: access denied -> next source"); continue
                raise
        else: raise RuntimeError("no accessible FLUX.1-schnell source")
        _flux.enable_model_cpu_offload(); _flux.vae.enable_slicing(); _flux.vae.enable_tiling()
        log("FLUX ready")
    return _flux

def gen_image(j, out):
    a = j["input"]["args"]; p = flux()
    g = torch.Generator("cpu").manual_seed(int(a.get("seed", 0)))
    im = p(prompt=a["prompt"], width=int(a.get("width", 1024)), height=int(a.get("height", 576)), num_inference_steps=int(a.get("num_inference_steps", 4)), guidance_scale=0.0, generator=g, max_sequence_length=256).images[0]
    if im.getextrema() == ((0, 0), (0, 0), (0, 0)): raise RuntimeError("black image (fp16 overflow?)")
    out.parent.mkdir(parents=True, exist_ok=True); im.save(out); return {"model": "FLUX.1-schnell nf4", "seed": a.get("seed")}

# ---------------------------------------------------------------- LTX-Video image->video (Lightricks 2B distilled), two-phase to fit Kaggle RAM/VRAM
LTX_ID = "Lightricks/LTX-Video"; LTX_W, LTX_H, LTX_FPS, LTX_MAXS = 768, 416, 24, 3.0   # 2B model (0.9.7-distilled is 13B: does not fit a T4)
LTX_STEPS, LTX_CFG = 50, 4.0
_ltx = None; _emb = {}
def ltx_encode_all(video_jobs):
    """phase 1: T5-XXL alone on the GPU -> prompt embeddings for every shot, then free it."""
    import gc
    from transformers import T5EncoderModel, T5TokenizerFast
    log("LTX phase 1: text encoder ...")
    tok = T5TokenizerFast.from_pretrained(LTX_ID, subfolder="tokenizer")
    enc = T5EncoderModel.from_pretrained(LTX_ID, subfolder="text_encoder", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
    def embed(text):
        t = tok(text, padding="max_length", max_length=128, truncation=True, add_special_tokens=True, return_tensors="pt")
        with torch.no_grad(): e = enc(t.input_ids.to(DEV), attention_mask=t.attention_mask.to(DEV)).last_hidden_state
        m = t.attention_mask.to(DEV)
        return (e * m.unsqueeze(-1).to(e.dtype)).cpu(), m.cpu()
    for j in video_jobs:
        a = j["input"]["args"]; _emb[j["id"]] = (embed(a["prompt"]), embed(a.get("negative_prompt", "")))
    del enc; gc.collect(); torch.cuda.empty_cache(); log(f"  embeddings for {len(_emb)} shots; text encoder freed")

def ltx():
    global _ltx
    if _ltx is None:
        from diffusers import LTXImageToVideoPipeline
        log("LTX phase 2: transformer + vae ...")
        _ltx = LTXImageToVideoPipeline.from_pretrained(LTX_ID, text_encoder=None, tokenizer=None, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(DEV)
        _ltx.vae.enable_tiling(); log("LTX ready")
    return _ltx

def gen_video(j, out):
    from diffusers.utils import export_to_video
    a = j["input"]["args"]; p = ltx()
    img_rel = a["input_image_filepath"].replace("asset:", "")
    src = GEN / img_rel if (GEN / img_rel).exists() else prev_gen / img_rel
    if not src.exists(): raise FileNotFoundError(f"keyframe missing: {img_rel}")
    img = Image.open(src).convert("RGB").resize((LTX_W, LTX_H))
    secs = min(LTX_MAXS, float(a.get("duration_ui", 2))); frames = (int(secs * LTX_FPS) // 8) * 8 + 1
    (pe, pm), (ne, nm) = _emb[j["id"]]
    g = torch.Generator("cpu").manual_seed(int(a.get("seed_ui", 42))); torch.cuda.empty_cache()
    vid = p(image=img, prompt_embeds=pe.to(DEV), prompt_attention_mask=pm.to(DEV), negative_prompt_embeds=ne.to(DEV), negative_prompt_attention_mask=nm.to(DEV),
            width=LTX_W, height=LTX_H, num_frames=frames, num_inference_steps=LTX_STEPS, guidance_scale=LTX_CFG, generator=g, decode_timestep=0.03, decode_noise_scale=0.025).frames[0]
    out.parent.mkdir(parents=True, exist_ok=True); export_to_video(vid, str(out), fps=LTX_FPS); return {"model": LTX_ID, "frames": frames, "fps": LTX_FPS, "size": [LTX_W, LTX_H]}

# ---------------------------------------------------------------- run
fails = 0
vj = [j for j in pending if j["type"] == "video"]
if vj:
    try: ltx_encode_all(vj)
    except Exception as e: log("LTX phase 1 failed:", e); traceback.print_exc()
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

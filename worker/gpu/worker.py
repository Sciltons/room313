#!/usr/bin/env python3
"""
Room 313 GPU worker for a 24 GB+ card with bf16 (Camber L4, RunPod, etc.). Runs inside pkg/:
  pkg/job.json {project, stage, only}, pkg/manifest.json, pkg/generated/** (existing assets), pkg/gpu/worker.py
Writes pkg/out/generated/** + pkg/out/gpu_status.json. FLUX.1-schnell (bf16) for images, LTX-Video 0.9.7 13B distilled for i2v.
"""
import os, sys, json, time, pathlib, traceback, gc
HERE = pathlib.Path(__file__).resolve().parent; PKG = HERE.parent
JOB = json.loads((PKG/"job.json").read_text()); MAN = json.loads((PKG/"manifest.json").read_text())
OUT = PKG/"out"/"generated"; OUT.mkdir(parents=True, exist_ok=True); PREV = PKG/"generated"
ASSETS = pathlib.Path(os.environ.get("ASSETS_DIR") or (PKG.parent/"assets")); ASSETS.mkdir(parents=True, exist_ok=True)   # persistent on the stash
STATUS = {}; T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
def save():
    (PKG/"out"/"gpu_status.json").write_text(json.dumps(STATUS, indent=1)); (ASSETS/"gpu_status.json").write_text(json.dumps(STATUS, indent=1))

stage, only = JOB.get("stage","video"), set(JOB.get("only", []))
def in_stage(j): return stage == "all" or j["type"] == stage or j.get("stage") == stage or stage in j.get("tags", []) or (stage == "images" and j["type"] == "image")
jobs = [j for j in MAN["jobs"] if in_stage(j) and (not only or j["id"] in only) and j.get("provider") == "hf_space"]
import shutil
pending = [j for j in jobs if j["id"] in only or not ((PREV/j["out"]).exists() or (ASSETS/j["out"]).exists())]
for j in jobs:   # resume: assets from an earlier (failed) run on the stash count as done and are re-exported
    if (ASSETS/j["out"]).exists() and not (OUT/j["out"]).exists(): (OUT/j["out"]).parent.mkdir(parents=True, exist_ok=True); shutil.copy(ASSETS/j["out"], OUT/j["out"])
import torch
log(f"stage={stage} jobs={len(jobs)} pending={len(pending)} gpu={torch.cuda.get_device_name(0)} vram={torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB bf16={torch.cuda.is_bf16_supported()}")
if not pending: save(); sys.exit(0)
from PIL import Image
DEV = "cuda"; BF = torch.bfloat16

# ---------------------------------------------------------------- FLUX.1-schnell bf16 (fits 24 GB with cpu offload of T5)
_flux = None
def flux():
    global _flux
    if _flux is None:
        from diffusers import FluxPipeline, FluxTransformer2DModel
        mid = os.environ.get("FLUX_REPO") or "YuCollection/FLUX.1-schnell-Diffusers"   # unmodified Apache-2.0 mirror (official repo is gated)
        log(f"loading {mid} (fp8 storage / bf16 compute) ...")
        tr = FluxTransformer2DModel.from_pretrained(mid, subfolder="transformer", torch_dtype=BF)
        tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=BF)   # 12B params -> ~12 GB, fits a 22 GB L4
        _flux = FluxPipeline.from_pretrained(mid, transformer=tr, torch_dtype=BF); _flux.enable_model_cpu_offload(); _flux.vae.enable_tiling()
        log(f"FLUX ready vram={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return _flux
def gen_image(j, out):
    a = j["input"]["args"]; p = flux(); g = torch.Generator("cpu").manual_seed(int(a.get("seed", 0)))
    im = p(prompt=a["prompt"], width=int(a.get("width",1024)), height=int(a.get("height",576)), num_inference_steps=int(a.get("num_inference_steps",4)), guidance_scale=0.0, generator=g, max_sequence_length=256).images[0]
    out.parent.mkdir(parents=True, exist_ok=True); im.save(out); return {"model": "FLUX.1-schnell bf16", "seed": a.get("seed")}

# ---------------------------------------------------------------- LTX-Video 0.9.7 13B distilled (bf16 weights ~26 GB -> fp8 storage via layerwise casting, bf16 compute)
LTX_ID = "Lightricks/LTX-Video-0.9.7-distilled"; LTX_W, LTX_H, LTX_FPS, LTX_MAXS = 896, 512, 24, 3.0
DISTILLED_T = [1000, 993, 987, 981, 975, 909, 725, 0.03]
_ltx = None; _emb = {}
def ltx_encode_all(video_jobs):
    from transformers import T5EncoderModel, T5TokenizerFast
    log("LTX phase 1: T5 text encoder ...")
    tok = T5TokenizerFast.from_pretrained(LTX_ID, subfolder="tokenizer"); enc = T5EncoderModel.from_pretrained(LTX_ID, subfolder="text_encoder", torch_dtype=BF).to(DEV).eval()
    def embed(text):
        t = tok(text, padding="max_length", max_length=128, truncation=True, add_special_tokens=True, return_tensors="pt")
        with torch.no_grad(): e = enc(t.input_ids.to(DEV), attention_mask=t.attention_mask.to(DEV)).last_hidden_state
        m = t.attention_mask.to(DEV); return (e * m.unsqueeze(-1).to(e.dtype)).cpu(), m.cpu()
    for j in video_jobs:
        a = j["input"]["args"]; _emb[j["id"]] = (embed(a["prompt"]), embed(a.get("negative_prompt","")))
    del enc; gc.collect(); torch.cuda.empty_cache(); log(f"  {len(_emb)} prompt embeddings; T5 freed")
def ltx():
    global _ltx
    if _ltx is None:
        from diffusers import LTXConditionPipeline, LTXVideoTransformer3DModel
        log("LTX phase 2: 13B transformer (fp8 storage / bf16 compute) + vae ...")
        tr = LTXVideoTransformer3DModel.from_pretrained(LTX_ID, subfolder="transformer", torch_dtype=BF)
        tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=BF)
        _ltx = LTXConditionPipeline.from_pretrained(LTX_ID, transformer=tr, text_encoder=None, tokenizer=None, torch_dtype=BF).to(DEV)
        _ltx.scheduler.register_to_config(use_dynamic_shifting=False, shift=1.0); _ltx.vae.enable_tiling()
        log(f"LTX ready vram={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return _ltx
def gen_video(j, out):
    from diffusers.utils import export_to_video
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
    a = j["input"]["args"]; p = ltx()
    rel = a["input_image_filepath"].replace("asset:", ""); src = next((d/rel for d in (OUT, ASSETS, PREV) if (d/rel).exists()), PREV/rel)
    if not src.exists(): raise FileNotFoundError(f"keyframe missing: {rel}")
    img = Image.open(src).convert("RGB").resize((LTX_W, LTX_H))
    secs = min(LTX_MAXS, float(a.get("duration_ui", 2))); frames = (int(secs*LTX_FPS)//8)*8 + 1
    (pe, pm), (ne, nm) = _emb[j["id"]]; g = torch.Generator("cpu").manual_seed(int(a.get("seed_ui", 42))); t0 = time.time()
    vid = p(conditions=[LTXVideoCondition(image=img, frame_index=0)], prompt_embeds=pe.to(DEV, BF), prompt_attention_mask=pm.to(DEV), negative_prompt_embeds=ne.to(DEV, BF), negative_prompt_attention_mask=nm.to(DEV),
            width=LTX_W, height=LTX_H, num_frames=frames, timesteps=DISTILLED_T, guidance_scale=1.0, guidance_rescale=0.7, image_cond_noise_scale=0.0,
            decode_timestep=0.05, decode_noise_scale=0.025, generator=g).frames[0]
    out.parent.mkdir(parents=True, exist_ok=True); export_to_video(vid, str(out), fps=LTX_FPS)
    return {"model": LTX_ID, "pipeline": "LTXConditionPipeline", "frames": frames, "fps": LTX_FPS, "size": [LTX_W, LTX_H], "gen_s": round(time.time()-t0, 1)}

# ---------------------------------------------------------------- run
# order: images first (first result within ~1 min), then text encoding, then videos
vj = [j for j in pending if j["type"] == "video"]
pending = [j for j in pending if j["type"] != "video"] + vj
fails = 0; _enc_done = False
for j in pending:
    if j["type"] == "video" and not _enc_done:
        _enc_done = True
        img_ok = sum(1 for k,v in STATUS.items() if v["state"]=="ok" and not k.startswith("vid_")); img_fail = sum(1 for k,v in STATUS.items() if v["state"]!="ok" and not k.startswith("vid_"))
        if img_fail and not img_ok: log("ABORT: all image jobs failed, skipping videos"); fails += 1; break
        if _flux is not None: _flux = None; gc.collect(); torch.cuda.empty_cache()   # free FLUX before LTX (module-level name)
        try: ltx_encode_all(vj)
        except Exception as e: log("LTX phase 1 failed:", e); traceback.print_exc()
    out = OUT/j["out"]
    try:
        log(f"[{j['id']}] {j['type']} ...")
        m = gen_image(j, out) if j["type"] == "image" else gen_video(j, out) if j["type"] == "video" else None
        if m is None: STATUS[j["id"]] = {"state": "skipped"}; continue
        (ASSETS/j["out"]).parent.mkdir(parents=True, exist_ok=True); shutil.copy(out, ASSETS/j["out"])   # checkpoint immediately
        STATUS[j["id"]] = {"state": "ok", "out": j["out"], "metrics": m, "elapsed": round(time.time()-T0, 1)}; log(f"[{j['id']}] OK {m.get('gen_s','')}")
    except Exception as e:
        fails += 1; STATUS[j["id"]] = {"state": "error", "error": str(e)[:400]}; log(f"[{j['id']}] ERROR {e}"); traceback.print_exc()
    save(); torch.cuda.empty_cache()
log(f"done. ok={sum(1 for v in STATUS.values() if v['state']=='ok')} failed={fails}"); sys.exit(1 if fails else 0)

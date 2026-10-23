#!/usr/bin/env python3
"""storyboard.json -> manifest.json (generation jobs with dependencies, stages, cost estimates)."""
import sys, json, pathlib, re
ROOT = pathlib.Path(__file__).resolve().parents[1]

MODELS = {
    "t2i":   "black-forest-labs/flux-1.1-pro",
    "ref":   "black-forest-labs/flux-kontext-pro",     # reference-conditioned (character consistency)
    "i2v":   "kwaivgi/kling-v2.1",
    "audio": "stackadoc/stable-audio-open-1.0",
}
COST = {"t2i": 0.04, "ref": 0.04, "i2v": 0.55, "audio": 0.02}

def fill(txt, sb):
    def rep(m):
        k = m.group(1)
        if k in sb["environments"]: return sb["environments"][k]
        if k in sb["characters"]: return sb["characters"][k]["prompt"]
        return m.group(0)
    return re.sub(r"\{(\w+)\}", rep, txt)

FREE = {
    "image_space": {"space": "black-forest-labs/FLUX.1-schnell", "api_name": "/infer"},          # Apache-2.0 weights, free Space quota
    "video_space": {"space": "Lightricks/ltx-video-distilled", "api_name": "/image_to_video"},   # verified/adjusted by free-audit (spaces_api.txt)
}
def build_free(project):
    proj = ROOT/"projects"/project; sb = json.loads((proj/"storyboard.json").read_text()); st = sb["style"]; jobs = []
    seed_base = 313
    for i, (cid, c) in enumerate(sb["characters"].items()):
        jobs.append({"id": f"char_{cid}", "type": "image", "stage": "character", "provider": "hf_space", "model": "FLUX.1-schnell@space", "est_usd": 0,
            "input": {**FREE["image_space"], "args": {"prompt": f"character reference sheet, front view and three-quarter view side by side, full body, neutral dark grey background, {c['prompt']}, {st['positive']}",
                      "seed": seed_base + i, "randomize_seed": False, "width": 1024, "height": 576, "num_inference_steps": 4}},
            "out": f"characters/{cid}.png"})
    for s_ in sb["shots"]:
        prompt = fill(s_["image"], sb) + ", " + st["positive"]
        # consistency without a paid reference model: identical character description block + per-character fixed seed
        seed = seed_base + (list(sb["characters"]).index(s_["chars"][0]) if s_["chars"] else 99)
        jobs.append({"id": f"kf_{s_['id']}", "type": "image", "stage": "image", "provider": "hf_space", "model": "FLUX.1-schnell@space", "est_usd": 0,
            "input": {**FREE["image_space"], "args": {"prompt": prompt, "seed": seed, "randomize_seed": False, "width": 1024, "height": 576, "num_inference_steps": 4}},
            "out": f"keyframes/{s_['id']}.png"})
    for s_ in sb["shots"]:
        jobs.append({"id": f"vid_{s_['id']}", "type": "video", "stage": "video", "provider": "hf_space", "model": "LTX-Video-distilled@space", "est_usd": 0, "expect_duration": 3,
            "input": {**FREE["video_space"], "args": {"prompt": fill(s_["motion"], sb) + ", " + st["motion_default"], "negative_prompt": st["negative"], "image": f"asset:keyframes/{s_['id']}.png"}},
            "out": f"animated_shots/{s_['id']}.mp4"})
    for mid, m in sb["audio_assets"]["music"].items():
        jobs.append({"id": f"music_{mid}", "type": "audio", "stage": "audio", "provider": "local_cpu", "model": "musicgen-small@cpu", "est_usd": 0, "expect_duration": min(30, m["duration"]),
            "input": {"kind": "musicgen", "prompt": m["prompt"], "duration": min(30, m["duration"])}, "out": f"audio/music/{mid}.wav"})
    for sid, x in sb["audio_assets"]["sfx"].items():
        jobs.append({"id": f"sfx_{sid}", "type": "audio", "stage": "audio", "provider": "freesound", "model": "freesound-cc0", "est_usd": 0,
            "input": {"query": x["prompt"].split(",")[0], "min_dur": 0.5, "max_dur": max(8, x["duration"] * 3)}, "out": f"audio/sfx/{sid}.mp3"})
    (proj/"manifest.json").write_text(json.dumps({"project": project, "mode": "zero-cost", "jobs": jobs}, indent=1, ensure_ascii=False))
    print(f"{len(jobs)} zero-cost jobs written (paid providers: none)")

def build(project, sfx_provider="replicate"):
    proj = ROOT/"projects"/project; sb = json.loads((proj/"storyboard.json").read_text())
    st = sb["style"]; jobs = []
    # 1) character reference sheets
    for cid, c in sb["characters"].items():
        jobs.append({"id": f"char_{cid}", "type": "image", "stage": "character", "model": MODELS["t2i"], "est_usd": COST["t2i"],
            "input": {"prompt": f"character reference sheet, front view and three-quarter view side by side, full body, neutral dark grey background, {c['prompt']}, {st['positive']}",
                      "aspect_ratio": "16:9", "output_format": "png", "safety_tolerance": 2, "prompt_upsampling": False},
            "out": f"characters/{cid}.png"})
    # 2) keyframes (reference-conditioned when characters appear)
    for s in sb["shots"]:
        prompt = fill(s["image"], sb) + ", " + st["positive"]
        if s["chars"]:
            ref = s["chars"][0]
            jobs.append({"id": f"kf_{s['id']}", "type": "image", "stage": "image", "model": MODELS["ref"], "est_usd": COST["ref"],
                "input": {"prompt": f"Place this exact character (same face, hair, clothes) into the scene: {prompt}", "input_image": f"asset:characters/{ref}.png",
                          "aspect_ratio": "16:9", "output_format": "png", "safety_tolerance": 2},
                "out": f"keyframes/{s['id']}.png"})
        else:
            jobs.append({"id": f"kf_{s['id']}", "type": "image", "stage": "image", "model": MODELS["t2i"], "est_usd": COST["t2i"],
                "input": {"prompt": prompt, "aspect_ratio": "16:9", "output_format": "png", "safety_tolerance": 2, "prompt_upsampling": False},
                "out": f"keyframes/{s['id']}.png"})
    # 3) image-to-video
    for s in sb["shots"]:
        dur = s["t"][1]-s["t"][0]; vd = 10 if dur > 5.0 else 5
        jobs.append({"id": f"vid_{s['id']}", "type": "video", "stage": "video", "model": MODELS["i2v"], "est_usd": COST["i2v"], "units": vd//5, "expect_duration": vd,
            "input": {"start_image": f"asset:keyframes/{s['id']}.png", "prompt": fill(s["motion"], sb) + ", " + st["motion_default"], "negative_prompt": st["negative"],
                      "duration": vd, "mode": "standard", "aspect_ratio": "16:9"},
            "out": f"animated_shots/{s['id']}.mp4"})
    # 4) music + sfx (non-speech only)
    for mid, m in sb["audio_assets"]["music"].items():
        jobs.append({"id": f"music_{mid}", "type": "audio", "stage": "audio", "model": MODELS["audio"], "est_usd": COST["audio"], "expect_duration": m["duration"],
            "input": {"prompt": m["prompt"], "seconds_total": min(47, m["duration"]), "steps": 100, "cfg_scale": 6}, "out": f"audio/music/{mid}.wav"})
    for sid, x in sb["audio_assets"]["sfx"].items():
        if sfx_provider == "elevenlabs":
            jobs.append({"id": f"sfx_{sid}", "type": "audio", "stage": "audio", "provider": "elevenlabs_sfx", "model": "elevenlabs_sfx", "est_usd": 0.02, "expect_duration": x["duration"],
                "input": {"prompt": x["prompt"], "duration": x["duration"]}, "out": f"audio/sfx/{sid}.mp3"})
        else:
            jobs.append({"id": f"sfx_{sid}", "type": "audio", "stage": "audio", "model": MODELS["audio"], "est_usd": COST["audio"], "expect_duration": x["duration"],
                "input": {"prompt": x["prompt"] + ", sound effect, no music", "seconds_total": min(47, x["duration"]), "steps": 100, "cfg_scale": 6}, "out": f"audio/sfx/{sid}.wav"})
    man = {"project": project, "jobs": jobs}
    (proj/"manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    by = {}
    for j in jobs: by[j["stage"]] = by.get(j["stage"], 0) + j["est_usd"]*j.get("units", 1)
    print(f"{len(jobs)} jobs; estimated cost by stage:", {k: round(v, 2) for k, v in by.items()}, "total ~$%.2f" % sum(by.values()))

if __name__ == "__main__":
    (build if len(sys.argv) > 2 and sys.argv[2] == "paid" else build_free)(sys.argv[1])

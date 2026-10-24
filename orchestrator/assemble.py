#!/usr/bin/env python3
"""
Assemble the film: generated shots + narration (master clock) + music/ambience/sfx -> renders/final.mp4
  python3 orchestrator/assemble.py <project> [--placeholders]
--placeholders: builds grey test clips for missing shots (pipeline test only; clearly labelled in the QC report).
"""
import sys, json, subprocess, pathlib, shutil, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]
FONT = str(ROOT/"orchestrator"/"fonts"/"Cairo-Bold.ttf")
if not pathlib.Path(FONT).exists(): FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0: raise RuntimeError(f"cmd failed: {' '.join(map(str,cmd))[:300]}\n{r.stderr[-1500:]}")
    return r
def probe_dur(f):
    return float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",str(f)],capture_output=True,text=True).stdout.strip() or 0)
def esc(t): return t.replace("\\","\\\\").replace(":","\\:").replace("'","\u2019").replace("%","\\%")

CAP_HTML = """<!doctype html><html dir="rtl"><head><meta charset="utf-8"><style>
@font-face{font-family:Cairo;src:url("file://%s")} html,body{margin:0;width:1080px;height:1920px;background:transparent;overflow:hidden}
.t{font-family:Cairo,"DejaVu Sans",sans-serif;font-weight:700;position:absolute;left:0;right:0;text-align:center;direction:rtl;white-space:nowrap}
#tag{top:%dpx;color:#e6e0d4;font-size:34px;letter-spacing:3px} #tag span{display:inline-block;padding:10px 36px 14px;border:2px solid rgba(155,17,30,.9);border-radius:4px;background:rgba(11,11,13,.6)}
#cap{top:%dpx;color:%s;font-size:%dpx;line-height:1.4;padding:0 30px;text-shadow:0 3px 0 #000,0 0 22px rgba(0,0,0,.95)}
#stamp{top:%dpx;color:#b5121f;font-size:104px} #stamp span{display:inline-block;padding:0 40px 10px;border:8px solid #b5121f;border-radius:8px;transform:rotate(-6deg)}
</style></head><body><div id="tag" class="t"><span>غُرفة ٣١٣</span></div><div id="cap" class="t">%s</div>%s</body></html>"""
# ---------- procedural FX (ffmpeg blend with time expressions). Storyboard: "fx": [{"type": ..., ...}], boxes in keyframe px (1024x576)
def _radial(w, h, cx, cy, r, color, feather=1.0):
    from PIL import Image
    import numpy as np
    yy, xx = np.mgrid[0:h, 0:w]; d = np.sqrt((xx-cx)**2 + (yy-cy)**2) / max(r, 1)
    a = np.clip(1 - d, 0, 1) ** (1.6 * feather); img = np.zeros((h, w, 3), np.uint8)
    for i, c in enumerate(color): img[..., i] = (a * c).astype(np.uint8)
    return Image.fromarray(img)
def _box(w, h, x0, y0, bw, bh, color=(255,255,255)):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (0,0,0)); ImageDraw.Draw(im).rectangle([x0, y0, x0+bw, y0+bh], fill=color); return im
def fx_chain(shot, fr, d, work):
    """returns (extra ffmpeg inputs, filter snippet turning [sh] into [shfx])"""
    inputs, chain, cur, n = [], [], "sh", [1]   # input index 0 = clip, 1 = caption, then fx images
    sx, sy = fr["w"]/1024, fr["h"]/576; W, H = fr["w"], fr["h"]
    def add_img(im, tag):
        pth = work/f"{shot['id']}_fx_{tag}.png"; im.save(pth); inputs.extend(["-loop","1","-t",str(d),"-i",str(pth)]); n[0] += 1; return n[0]
    for k, fx in enumerate(shot.get("fx", [])):
        t = fx["type"]; st = fx.get("strength", 1.0)
        if t == "flicker":        # warm/red light flicker around a point: A + glow*k(T)
            cx, cy, r = fx["at"][0]*sx, fx["at"][1]*sy, fx.get("radius", 220)*sx; col = fx.get("color", [255,140,40])
            i = add_img(_radial(W, H, cx, cy, r, col), f"flicker{k}")
            expr = f"min(255,A+B*{0.35*st:.3f}*(0.5+0.5*sin(29*T)*sin(7.3*T+1)+0.3*sin(53*T)))"
            chain.append(f"[{i}:v]format=rgb24[fx{k}];[{cur}][fx{k}]blend=all_expr='{expr}'[sh{k}]"); cur = f"sh{k}"
        elif t == "pulse":        # whole-frame color pulse (e.g. red)
            col = fx.get("color", [255,0,0]); i = add_img(_box(W, H, 0, 0, W, H, tuple(col)), f"pulse{k}"); f = fx.get("hz", 1.2)
            expr = f"min(255,A+B*{0.25*st:.3f}*(0.5+0.5*sin(2*PI*{f}*T)))"
            chain.append(f"[{i}:v]format=rgb24[fx{k}];[{cur}][fx{k}]blend=all_expr='{expr}'[sh{k}]"); cur = f"sh{k}"
        elif t == "blink":        # darken boxes (eyes) briefly at given times
            im = _box(W, H, 0, 0, 1, 1)
            from PIL import ImageDraw
            dr = ImageDraw.Draw(im)
            for b in fx["boxes"]: dr.rectangle([b[0]*sx, b[1]*sy, (b[0]+b[2])*sx, (b[1]+b[3])*sy], fill=(255,255,255))
            i = add_img(im, f"blink{k}"); times = fx.get("times", [d*0.45]); dur = fx.get("dur", 0.14)
            gate = "+".join(f"between(T,{tm:.2f},{tm+dur:.2f})" for tm in times)
            chain.append(f"[{i}:v]format=rgb24[fx{k}];[{cur}][fx{k}]blend=all_expr='A*(1-B/255*{min(1,st):.2f}*({gate}))'[sh{k}]"); cur = f"sh{k}"
        elif t == "glint":        # bright spots (eyes) that glow and pulse
            im = _box(W, H, 0, 0, 1, 1)
            from PIL import Image
            import numpy as np
            arr = np.zeros((H, W, 3), np.float32)
            for b in fx["boxes"]:
                g = np.asarray(_radial(W, H, (b[0]+b[2]/2)*sx, (b[1]+b[3]/2)*sy, max(b[2], b[3])*sx*1.2, fx.get("color", [255,200,90]))).astype(np.float32); arr = np.maximum(arr, g)
            i = add_img(Image.fromarray(arr.astype(np.uint8)), f"glint{k}")
            chain.append(f"[{i}:v]format=rgb24[fx{k}];[{cur}][fx{k}]blend=all_expr='min(255,A+B*{0.9*st:.2f}*(0.55+0.45*sin(5.1*T)*sin(1.7*T)))'[sh{k}]"); cur = f"sh{k}"
        elif t == "pushin":       # slow zoom (works on video clips too)
            z = fx.get("zoom", 0.10)
            nf = int(d*30)
            chain.append(f"[{cur}]zoompan=z='1+{z}*on/{nf}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps=30[sh{k}]"); cur = f"sh{k}"
        elif t == "shake":
            amp = fx.get("amp", 6)
            chain.append(f"[{cur}]scale=iw*1.04:ih*1.04,crop=w={W}:h={H}:x='(iw-ow)/2+{amp}*sin(47*t)*sin(13*t)':y='(ih-oh)/2+{amp}*sin(31*t)'[sh{k}]"); cur = f"sh{k}"
    chain.append(f"[{cur}]null[shfx]")
    return inputs, ";".join(chain)

_pw=None
def caption_png(text, style, fr, out):
    global _pw
    from playwright.sync_api import sync_playwright
    if _pw is None:
        _pw=sync_playwright().start(); _pw.browser=_pw.chromium.launch(); _pw.page=_pw.browser.new_page(viewport={"width":1080,"height":1920})
    color="#e0303c" if style=="red" else "#f2ede4"; size=76 if style=="red" else 60
    stamp=f'<div id="stamp" class="t"><span>{text}</span></div>' if style=="stamp" else ""
    html=CAP_HTML%(FONT, fr["y"]-90, fr["y"]+fr["h"]+60, color, size, fr["y"]+fr["h"]//2-70, text, stamp)
    _pw.page.set_content(html); _pw.page.wait_for_timeout(50); _pw.page.screenshot(path=str(out), omit_background=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("project"); ap.add_argument("--placeholders",action="store_true"); a=ap.parse_args()
    proj=ROOT/"projects"/a.project; sb=json.loads((proj/"storyboard.json").read_text()); nar=json.loads((proj/"timeline"/"narration.json").read_text())
    gen=proj/"generated"; work=proj/"renders"/"work"; work.mkdir(parents=True,exist_ok=True); report={"shots":{}, "audio":{}, "placeholders":[]}
    W,H,FPS=sb["output"]["width"],sb["output"]["height"],sb["output"]["fps"]; fr=sb["output"]["frame"]
    total=max(nar["total"], sb["shots"][-1]["t"][1])

    # ---------- 1) per-shot clips, exact duration, framed into 9:16 with captions
    clips=[]
    for s in sb["shots"]:
        t0,t1=s["t"]; d=round(t1-t0,3); src=gen/"animated_shots"/f"{s['id']}.mp4"; kf=gen/"keyframes"/f"{s['id']}.png"; out=work/f"{s['id']}.mp4"
        if src.exists():
            sd=probe_dur(src); mode="video"
            # fit: slow down up to 1.6x, otherwise loop; trim to d
            if sd>=d: vf_fit=f"trim=0:{d},setpts=PTS-STARTPTS"
            elif sd*1.6>=d: vf_fit=f"setpts={d/sd:.4f}*PTS"
            else: vf_fit=f"loop=loop=-1:size={int(sd*FPS)},trim=0:{d},setpts=PTS-STARTPTS"
            inp=["-i",str(src)]
        elif kf.exists():
            mode="keyframe_kenburns"; report["placeholders"].append(f"{s['id']}: video missing, keyframe push-in used")
            inp=["-loop","1","-t",str(d),"-i",str(kf)]; vf_fit=f"scale=1600:-2,zoompan=z='min(zoom+0.0006,1.12)':d={int(d*FPS)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1376x774:fps={FPS}"
        elif a.placeholders:
            mode="PLACEHOLDER"; report["placeholders"].append(f"{s['id']}: PLACEHOLDER test clip")
            inp=["-f","lavfi","-t",str(d),"-i",f"color=c=0x2a2b30:s=1376x774:r={FPS}"]; vf_fit="null"
        else:
            raise SystemExit(f"missing shot {s['id']} (no video, no keyframe). Run generation first.")
        style=s.get("caption_style",""); capfile=work/f"{s['id']}_cap.png"; caption_png(s.get("caption",""), style, fr, capfile)
        flash = f",drawbox=x=0:y=0:w=iw:h=ih:color=0xff2a2a@0.5:t=fill:enable='lt(t,0.12)'" if style=="red" else ""
        fx_in, fx_f = fx_chain(s, fr, d, work)
        vf=(f"[0:v]{vf_fit},scale={fr['w']}:{fr['h']}:force_original_aspect_ratio=increase,crop={fr['w']}:{fr['h']},fps={FPS},format=rgb24[sh];{fx_f};[shfx]format=rgba[shr];"
            f"color=c=black:s={W}x{H}:r={FPS}:d={d}[bg];[bg][shr]overlay={fr['x']}:{fr['y']}:shortest=1[base];"
            f"[1:v]format=rgba,fade=t=in:st=0:d=0.25:alpha=1[cap];[base][cap]overlay=0:0:shortest=1{flash},"
            f"vignette=PI/4,noise=alls=12:allf=t+u,fade=t=in:st=0:d=0.08:color=black,fade=t=out:st={max(0,d-0.08)}:d=0.08:color=black,format=yuv420p[v]")
        run(["ffmpeg","-y","-loglevel","error",*inp,"-loop","1","-t",str(d),"-i",str(capfile),*fx_in,"-filter_complex",vf,"-map","[v]","-an","-t",str(d),"-r",str(FPS),"-c:v","libx264","-preset","veryfast","-crf","18",str(out)])
        got=probe_dur(out); clips.append(out); report["shots"][s["id"]]={"mode":mode,"target":d,"got":round(got,3)}
        print(f"{s['id']} {mode:18s} {d:5.2f}s -> {got:5.2f}s")

    # ---------- 2) video concat
    lst=work/"concat.txt"; lst.write_text("".join(f"file '{c}'\n" for c in clips))
    video=work/"video.mp4"; run(["ffmpeg","-y","-loglevel","error","-f","concat","-safe","0","-i",str(lst),"-c","copy",str(video)])

    # ---------- 3) audio: narration (untouched) + music beds + sfx, ducked under the voice
    inputs=[]; fc=[]; idx=0
    def add(path):
        nonlocal idx; inputs.extend(["-i",str(path)]); idx+=1; return idx-1
    voice=[]
    for f in nar["files"]:
        i=add(proj/"input"/f["file"]); fc.append(f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo,adelay={int(f['offset']*1000)}|{int(f['offset']*1000)},apad=whole_dur={total}[v{i}]"); voice.append(f"[v{i}]")
    fc.append(f"{''.join(voice)}amix=inputs={len(voice)}:normalize=0,highpass=f=80,acompressor=threshold=-16dB:ratio=3:attack=8:release=150,volume=1.4[voice]")
    # music: one bed per contiguous run of shots sharing the same music id
    beds=[]; runs=[]
    for s in sb["shots"]:
        m=s.get("music")
        if runs and runs[-1][0]==m: runs[-1][2]=s["t"][1]
        else: runs.append([m,s["t"][0],s["t"][1]])
    for k,(m,t0,t1) in enumerate(runs):
        p=gen/"audio"/"music"/f"{m}.wav"
        if not p.exists(): report["audio"][f"music_{m}"]="missing"; continue
        i=add(p); d=t1-t0; fade=min(1.2,d/3)
        fc.append(f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo,aloop=loop=-1:size=2e9,atrim=0:{d},afade=t=in:d={fade},afade=t=out:st={max(0,d-fade)}:d={fade},volume=0.55,adelay={int(t0*1000)}|{int(t0*1000)},apad=whole_dur={total}[m{k}]"); beds.append(f"[m{k}]")
    # sfx at shot start
    sfxs=[]; n=0
    for s in sb["shots"]:
        for j,sid in enumerate(s.get("sfx",[])):
            p=next(iter((gen/"audio"/"sfx").glob(f"{sid}.*")),None)
            if not p: report["audio"][f"sfx_{sid}"]="missing"; continue
            i=add(p); at=s["t"][0]+0.05*j; d=s["t"][1]-s["t"][0]
            fc.append(f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo,atrim=0:{d+1.5},afade=t=out:st={d},d=1.5,volume=0.7,adelay={int(at*1000)}|{int(at*1000)},apad=whole_dur={total}[s{n}]"); sfxs.append(f"[s{n}]"); n+=1
    bedlist=beds+sfxs
    if bedlist:
        fc.append(f"{''.join(bedlist)}amix=inputs={len(bedlist)}:normalize=0[bedraw];[voice]asplit[vA][vB];[bedraw][vB]sidechaincompress=threshold=-30dB:ratio=3:attack=15:release=250[bed];[vA][bed]amix=inputs=2:normalize=0,alimiter=limit=0.97,loudnorm=I=-14:TP=-1.2:LRA=10[a]")
    else:
        fc.append("[voice]alimiter=limit=0.97,loudnorm=I=-14:TP=-1.2:LRA=10[a]")
    audio=work/"audio.m4a"
    run(["ffmpeg","-y","-loglevel","error",*inputs,"-filter_complex",";".join(fc),"-map","[a]","-t",str(total),"-c:a","aac","-b:a","192k",str(audio)])

    # ---------- 4) mux + validate
    final=proj/"renders"/"final.mp4"
    run(["ffmpeg","-y","-loglevel","error","-i",str(video),"-i",str(audio),"-map","0:v","-map","1:a","-c:v","copy","-c:a","copy","-shortest","-movflags","+faststart",str(final)])
    vd=probe_dur(final); streams=subprocess.run(["ffprobe","-v","error","-show_entries","stream=codec_type,codec_name,width,height","-of","csv=p=0",str(final)],capture_output=True,text=True).stdout.strip().split("\n")
    black=subprocess.run(["ffmpeg","-i",str(final),"-vf","blackdetect=d=0.4:pix_th=0.02","-an","-f","null","-"],capture_output=True,text=True).stderr
    report["final"]={"path":str(final),"duration":round(vd,3),"expected":round(total,3),"streams":streams,"black_segments":[l for l in black.splitlines() if "black_start" in l]}
    (proj/"quality_reports").mkdir(exist_ok=True); (proj/"quality_reports"/"assembly.json").write_text(json.dumps(report,indent=1,ensure_ascii=False))
    ok = abs(vd-total)<0.5 and len(streams)==2
    print("FINAL", final, f"{vd:.2f}s / expected {total:.2f}s", "OK" if ok else "CHECK", "| placeholders:", len(report["placeholders"]), "| missing audio:", len(report["audio"]))
    return 0 if ok else 1

if __name__=="__main__": sys.exit(main())

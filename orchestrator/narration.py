#!/usr/bin/env python3
"""Analyze narration files -> timeline/narration.json (segments, pauses, durations). The voice is the master clock."""
import sys, json, subprocess, pathlib, re
ROOT = pathlib.Path(__file__).resolve().parents[1]

def duration(f):
    return float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",str(f)],capture_output=True,text=True).stdout.strip())

def silences(f, db=-38, dmin=0.12):
    err = subprocess.run(["ffmpeg","-i",str(f),"-af",f"silencedetect=n={db}dB:d={dmin}","-f","null","-"],capture_output=True,text=True).stderr
    st = [float(x) for x in re.findall(r"silence_start: ([0-9.]+)", err)]
    en = [float(x) for x in re.findall(r"silence_end: ([0-9.]+)", err)]
    return list(zip(st, en))

def analyze(project, gap=0.4, lead=0.4):
    proj = ROOT/"projects"/project; files = sorted((proj/"input").glob("narration_*.mp3"))
    out = {"project": project, "lead_in": lead, "gap_between_files": gap, "files": [], "total": 0.0}
    cfg = json.loads((proj/"timeline"/"config.json").read_text()) if (proj/"timeline"/"config.json").exists() else {}
    cursor = 0.0
    for f in files:
        d = duration(f); sil = silences(f)
        segs=[]; prev=0.0
        for s,e in sil:
            if s-prev>0.05: segs.append([round(prev,3), round(s,3)])
            prev=e
        if d-prev>0.05: segs.append([round(prev,3), round(d,3)])
        start = cursor + lead
        out["files"].append({"file": f.name, "duration": round(d,3), "offset": round(start,3),
                             "speech": [[round(start+a,3), round(start+b,3)] for a,b in segs],
                             "pauses": [[round(start+a,3), round(start+b,3)] for a,b in sil]})
        cursor = start + d + cfg.get("gap_after",{}).get(f.name, gap)
    out["total"] = round(cursor,3)
    (proj/"timeline").mkdir(exist_ok=True); (proj/"timeline"/"narration.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out

if __name__ == "__main__":
    r = analyze(sys.argv[1])
    for f in r["files"]:
        print(f["file"], f["duration"], "offset", f["offset"]); print("  speech:", f["speech"])
    print("total", r["total"])

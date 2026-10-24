#!/usr/bin/env python3
"""Word-level timestamps for the narration (faster-whisper, CPU). Writes timeline/words.json: [{file, word, start, end}] in global timeline seconds."""
import sys, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
def main(project, model="medium"):
    from faster_whisper import WhisperModel
    proj = ROOT/"projects"/project; nar = json.loads((proj/"timeline"/"narration.json").read_text())
    m = WhisperModel(model, device="cpu", compute_type="int8"); words = []
    for f in nar["files"]:
        segs, info = m.transcribe(str(proj/"input"/f["file"]), language="ar", word_timestamps=True, vad_filter=False, beam_size=5)
        for s in segs:
            for w in (s.words or []): words.append({"file": f["file"], "word": w.word.strip(), "start": round(f["offset"]+w.start, 3), "end": round(f["offset"]+w.end, 3), "p": round(w.probability, 2)})
        print(f["file"], "words:", sum(1 for w in words if w["file"]==f["file"]), flush=True)
    (proj/"timeline"/"words.json").write_text(json.dumps(words, ensure_ascii=False, indent=0)); print("total words", len(words))
if __name__ == "__main__": main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "medium")

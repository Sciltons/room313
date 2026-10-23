# Room 313 Studio — zero-cost external-worker animation pipeline

```
story + narration (user)  ──►  orchestrator (Claude env)  ──►  GitHub Actions worker (free)  ──►  assets  ──►  assemble (ffmpeg)  ──►  final.mp4
                                 narration.py (voice = master clock)        worker/generate.py
                                 storyboard.json (shots, prompts, sfx)      hf_space  : public Hugging Face Spaces via gradio_client (FLUX.1-schnell images, LTX-Video i2v)
                                 build_manifest.py (jobs, $0)               local_cpu : open models on the runner CPU (musicgen-small music, LCM images)
                                 dispatch.py (run + poll + pull)            freesound : CC0 sound effects (free API key)
                                 assemble.py (captions, mix, validate)      resumable · ledger.json · status.json
```

**Cost policy: $0.** Paid providers (Replicate, fal, Kling, Runway, ElevenLabs SFX) are hard-disabled in `worker/generate.py`
(`ALLOW_PAID` must never be set). `build_manifest.py` writes zero-cost manifests by default; `python3 orchestrator/build_manifest.py <p> paid`
exists only as reference and produces jobs the worker refuses.

Optional **free** credentials (GitHub → Settings → Secrets and variables → Actions):
`HF_TOKEN` (free Hugging Face account — raises the Space quota a lot), `FREESOUND_API_KEY` (free, required for SFX), `KAGGLE_USERNAME`/`KAGGLE_KEY` (optional GPU worker).

Resume: jobs whose output exists are skipped; failed jobs are retried `max_retries` times, then reported in `status.json`.
Licence notes recorded per asset in `status.json`/`ledger.json`: musicgen-small weights are **CC-BY-NC** (non-commercial) — replace the music
before monetising; Freesound results are filtered to CC0 and the author is recorded.

## Run order
1. `python3 orchestrator/dispatch.py free-audit` — runner facts, which free Spaces are up and their real API signatures, one sample image.
   Results land in `quality_reports/audit_<run>/` (read `spaces_api.txt` first; fix `FREE["video_space"]` in `build_manifest.py` if LTX's api differs).
2. `python3 orchestrator/narration.py <project>` → `timeline/narration.json`
3. edit `projects/<project>/storyboard.json`, then `python3 orchestrator/build_manifest.py <project>`
4. **Milestone 0**: `python3 orchestrator/dispatch.py generate <project> milestone --dry-run false`
   → one character sheet + one keyframe + one animated shot (`generated/characters`, `keyframes`, `animated_shots`). Review before anything else.
5. stages `character` → `image` → `video` → `audio` (review between stages; `--only id1,id2` to redo single jobs)
6. `python3 orchestrator/assemble.py <project>` → `renders/final.mp4` + `quality_reports/assembly.json`
   (`--placeholders` builds grey test clips for missing shots — pipeline test only, clearly flagged in the report)

The generate workflow also fetches `orchestrator/fonts/Cairo-Bold.ttf` (Google Fonts, OFL) on first run and commits it, so Arabic captions
render with Cairo after the next `git pull`; until then `assemble.py` falls back to DejaVu Sans Bold.

# Room 313 Studio — external-worker animation pipeline

```
story + narration (user)  ──►  orchestrator (Claude env)  ──►  GitHub Actions worker  ──►  assets  ──►  assemble (ffmpeg)  ──►  final.mp4
                                 narration.py (voice = master clock)        worker/generate.py
                                 storyboard.json (shots, prompts, sfx)      Replicate: Flux (images) · Flux Kontext (character consistency)
                                 build_manifest.py (jobs + cost)                       Kling 2.1 (image→video) · Stable Audio (music/sfx)
                                 dispatch.py (run + poll + pull)            optional: ElevenLabs sound effects
                                 assemble.py (captions, mix, validate)      resumable · budget-capped · ledger.json
```

Secrets live only in GitHub → Settings → Secrets and variables → Actions:
`REPLICATE_API_TOKEN` (required), `ELEVENLABS_API_KEY` (optional).

Safety: `dry_run=true` by default; paid calls need `authorize=YES` **and** stay under `budget_usd` (cumulative in `projects/<p>/ledger.json`).
Resume: jobs whose output exists are skipped; failed jobs are retried `max_retries` times, then reported in `status.json`.

## Run order
1. `python3 orchestrator/dispatch.py connectivity` — runner reachability test (no secrets).
2. `python3 orchestrator/narration.py <project>` → `timeline/narration.json`
3. edit `projects/<project>/storyboard.json`, then `python3 orchestrator/build_manifest.py <project>`
4. `python3 orchestrator/dispatch.py generate <project> character --dry-run true` (plan + cost)
5. `... --dry-run false --authorize YES --budget 0.3` → review `generated/characters/*.png`
6. stages `image` → `video` → `audio` (review between stages)
7. `python3 orchestrator/assemble.py <project>` → `renders/final.mp4` + `quality_reports/assembly.json`

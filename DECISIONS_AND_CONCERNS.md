# Decisions and concerns

Morning report for the autonomous run. Newest entries at the bottom.

## [2026-11-16 01:05] Hardware is smaller than the brief assumes
- **Situation:** RTX 3060 Laptop (6 GB), 15 GB RAM, Windows.
- **Decision / action taken:** Plan is CNN + MERT only; AST dropped per user. DataLoader workers capped at 6 (not cores-2); CNN batch 32 with fallback 16.
- **Why:** AST fine-tuning would take 10-20 h; spawn workers cost ~0.5 GB RAM each.
- **How to change it:** `train.num_workers` in configs/*.yaml.
- **Concern level:** low

## [2026-11-16 01:05] transformers pinned below 5 for MERT
- **Situation:** MERT's remote code returned `hidden_states=None` under transformers 5.23.
- **Decision / action taken:** Pinned `transformers>=4.40,<5` (4.57.6), `huggingface-hub<1`, `gradio>=5,<6`, and pinned the MERT code revision in configs/mert.yaml. Verified 13 layers, no NaNs, fp16 matches fp32 (cos 0.9999998).
- **Why:** Needed for extraction to work at all.
- **How to change it:** requirements.txt; mert.revision in configs/mert.yaml.
- **Concern level:** low

## [2026-11-16 01:05] Single decode pass, files and design choices
- **Situation:** User chose one MP3 decode writing both caches.
- **Decision / action taken:** `preprocess_mels.py` writes data/mels (float16 128x1292) and data/wav24k (float16, 720000 samples). Tracks under 10 s of audio are logged to bad_tracks.txt; shorter-than-30 s but >=10 s are zero-padded. Splits/genres/stats are per subset (`splits_small.csv`, `mel_stats_medium.json`) rather than the brief's single `splits.csv`. Eval uses 5 windows of 10 s (5 s hop). Mel stats are global mean/std over training split. Class weighting: inv-sqrt CE weights (not a sampler). Gain augmentation is a dB offset applied before normalization. MERT head's LayerNorm is optional and off.
- **Why:** Matches the user's decisions; fills gaps the brief left open.
- **How to change it:** configs/*.yaml; src/preprocess_mels.py `MIN_SECONDS`.
- **Concern level:** low

## [2026-11-16 01:05] Brief's subset filter pitfall avoided
- **Situation:** `('set','subset') <= 'medium'` on plain pandas strings would include 'large'.
- **Decision / action taken:** Use `.isin(['small','medium'])` -> 25,000 tracks.
- **How to change it:** src/prepare_metadata.py
- **Concern level:** low

## [2026-11-16 01:25] 6 fma_small tracks failed and were logged as bad
- **Situation:** Preprocessing logged 6 bad tracks in data/bad_tracks.txt: 98565, 98567, 98569 (libsndfile decode error, corrupt MP3s) and 99134, 108925, 133297 (file missing from the archive).
- **Decision / action taken:** Excluded them everywhere (`load_splits` drops ids in bad_tracks.txt). 7,994 of 8,000 small tracks used.
- **Why:** The brief says to skip and log bad files. These match FMA's known-bad list.
- **How to change it:** Delete lines from data/bad_tracks.txt (not recommended).
- **Concern level:** low

## [2026-11-16 01:25] Windows cp1252 bug in YAML loading
- **Situation:** configs/cnn.yaml contains a non-ASCII character; Python read it as cp1252 and crashed.
- **Decision / action taken:** All text reads/writes in utils.py/train.py now use UTF-8.
- **How to change it:** src/utils.py `load_config`
- **Concern level:** low

## [2026-11-16 01:25] fma_small cache verified
- **Situation:** Milestone 2 check.
- **Decision / action taken:** Plotted outputs/figures/sample_small.png (mel, augmented crop, 16 kHz resample): looks correct. Training-split mel stats: mean -12.07 dB, std 17.13. Smoke test passed: batch 32 uses 1.66 GB GPU memory in bf16 (no OOM fallback needed).
- **Concern level:** low

## [2026-11-16 01:45] Pre-flight tests passed; overnight chain launched
- **Situation:** CNN quick run on fma_small (3 epochs): val macro-F1 0.265 -> 0.369 -> 0.441, ~25 s/epoch after a ~137 s first epoch (cudnn benchmark + worker start). MERT extraction test on 64 tracks: ~15 tracks/s, 3.4 GB peak GPU, finite (13, 768) outputs.
- **Decision / action taken:** Launched scripts/run_overnight.sh (log: logs/overnight.log) once the fma_medium extraction finishes: cache medium -> CNN full -> evaluate -> MERT extract -> MERT quick -> MERT full -> final eval with UMAP. Smoke tests ran only for the CNN; the MERT head's training path is first exercised by the chain's `--quick` step (this is its pre-flight).
- **Why:** The user asked for a quick test before every full run and for continuous progress. The 64 test embeddings in data/embeddings are valid and reused by the resumable extractor.
- **How to change it:** Edit scripts/run_overnight.sh; rerun with `bash scripts/run_overnight.sh > logs/overnight.log 2>&1 &`.
- **Concern level:** low
- **Note:** Pop recall was 0.05 after 3 epochs (underfit, expected this early); watch per-genre recall in the full run.

## [2026-11-16 02:40] Medium cache done; memory pressure killed a watcher task
- **Situation:** fma_medium cache finished (43 min): 24,980 tracks, 20 bad in total (all known-missing or corrupt files in the FMA archive, e.g. the 98xxx range). CNN full training began 02:33 (622 steps/epoch, ~127 s/epoch, slower than the 80 s I estimated, so ~65 min for 30 epochs). While the preprocessing pool (14 workers, ~8.7 GB) ran, Claude Code killed one of my idle wait-watcher tasks for low system memory (free RAM was 2.4 GB). The chain itself was not affected.
- **Decision / action taken:** Did not restart that watcher on my own. The user asked to resume; I confirmed from logs the chain is still running and continue by checking logs directly. Shell commands briefly failed with a transient classifier error, so I used read-only tools meanwhile.
- **Why:** Memory is tight on a 15 GB machine; the 14-worker default in preprocess_mels.py is above the 6-worker cap I use elsewhere.
- **How to change it:** `--workers N` on src/preprocess_mels.py (edit scripts/run_overnight.sh); `train.num_workers` in configs. The cache step is finished, so it no longer matters for this run.
- **Concern level:** medium (RAM headroom when training workers and MERT extraction overlap; the chain runs them one after another, which avoids it)
- **Early signal:** epoch 1-2 val macro-F1 0.282 / 0.288 on 16 classes (val acc 0.56 reflects the Rock/Electronic majority). Low but expected at peak LR; will judge by the final curve.

## [2026-11-16 02:43] Restarted CNN training with lighter validation workers
- **Situation:** Attempt 1 (logs/overnight_attempt1.log) had 14 python processes / ~8.75 GB: 6 train workers + 6 *persistent* validation workers + main. Free RAM fell to ~2.5 GB, epoch time drifted 127 s -> 186 s, GPU utilization was ~26%, and Claude Code killed an idle watcher task for memory pressure. Stopped at epoch 3 (val macro-F1 0.288, nothing worth keeping).
- **Decision / action taken:** In src/train.py the validation DataLoader now uses at most 2 workers and is not persistent (`train.val_num_workers`, default 2). Killed only my own chain/training processes, then relaunched scripts/run_overnight.sh (log: logs/overnight.log). Free RAM went from 2.5 GB to 9.3 GB after the stop. About 10 minutes of training lost.
- **Why:** Safer overnight on a 15 GB machine; a crash from memory exhaustion would cost far more than 10 minutes.
- **How to change it:** `train.val_num_workers` and `train.num_workers` in configs/cnn.yaml.
- **Concern level:** medium (laptop thermals: GPU read 85 C during training; sustained throttling would lengthen the run)

## [2026-11-16 04:00] Model A (CNN) finished; results
- **Situation:** Full CNN run completed all 30 epochs without early stopping (last restart: 02:44-03:56). Best val macro-F1 0.4635 at epoch 26 (val acc 0.713). Official test split (2,572 tracks, track level, 5 windows averaged): top-1 0.671, top-3 0.845, macro-F1 0.460, weighted-F1 0.651. Test is within 0.003 of validation macro-F1, so no sign of leakage or a split bug.
- **Per-genre (test F1):** Old-Time/Historic 0.94, Electronic 0.80, Rock 0.81, Hip-Hop 0.77, Classical 0.77, International 0.60, Jazz 0.54, Spoken 0.48, Experimental 0.42, Folk 0.36, Instrumental 0.34, Pop 0.30, Country 0.26, and **0.00 for Blues (8 test tracks), Easy Listening (6) and Soul-RnB (42)**. Reports: outputs/cnn/classification_report_test.txt, confusion_test.png, eval_test.json.
- **Decision / action taken:** Left as is. Did not tune on the test set. The zero-F1 genres are a known consequence of tiny training counts (Easy Listening 13, Blues 58, Soul-RnB 94 training tracks) with only inv-sqrt class weighting; Soul-RnB is the surprising one given 42 test tracks, so it will be looked at in the error analysis (likely absorbed by Hip-Hop/Electronic/Rock).
- **How to change it:** `train.class_weighting` (e.g. `inv`) or a weighted sampler in configs/cnn.yaml, then rerun train.py + evaluate.py.
- **Concern level:** low-medium (macro-F1 0.46 is modest but plausible for a from-scratch CNN on this imbalanced data; MERT is expected to beat it)

## [2026-11-16 04:00] Memory watcher killed a second time
- **Situation:** A second lightweight wait-watcher (task bo5y1zn2f) was stopped by Claude Code for low system memory (free RAM ~3 GB while MERT extraction runs). The chain (started with nohup) was not affected and is now on MERT extraction.
- **Decision / action taken:** Did not restart the watcher (per the notice). I check progress by reading logs/overnight.log directly instead.
- **How to change it:** Close other applications to free RAM, or start Claude Code with CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1.
- **Concern level:** medium

## [2026-11-16 10:55] Chain finished at 04:44; MERT (Model C) results, report, demo, README
- **Situation:** The overnight chain completed all steps by 04:44 (MERT extraction 03:57-04:39, MERT quick + full training, final eval with UMAP). I only noticed at 10:54 because the memory killer had stopped both of my watchers, so about 6 hours of idle time on my side (no cost to results).
- **Results (official test, 2,572 tracks):** MERT top-1 0.682, top-3 0.846, macro-F1 0.493, weighted-F1 0.680; CNN 0.671 / 0.845 / 0.460 / 0.651. MERT head: early-stopped at epoch 47, best val macro-F1 0.544 at epoch 32 (val is optimistic vs test; train acc ~0.79 vs val ~0.71 = head overfits). Learned layer weights peak at layer 6 (0.33).
- **Decision / action taken:** Wrote the error analysis into reports/results.md (confusions, rare-genre failure, caveats: one seed, no confidence intervals, 0.033 macro-F1 gap may be within noise). Tested app/demo.py live through the Gradio API: auto-picks the best evaluated run (mert), correct top-3 on a Hip-Hop track, a 4 s clip is padded without error. Wrote README.md with results, commands, design notes, an "Adding AST" guide, and lessons.
- **Concerns:** (1) Blues/Easy Listening/Soul-RnB are at ~0 F1 in both models (known, documented). (2) Soul-RnB -> Rock in 28/42 cases for both models. (3) MERT's edge over the CNN is modest and untested for significance. (4) Microphone input in the demo is untested (needs a browser; file upload via API was tested). (5) Measured sizes: mels 7.8 GB, wav24k 34 GB, embeddings 1 GB, raw 62 GB (README updated; an earlier 36 GB figure was a calculation).
- **How to change it:** Retrain with `train.class_weighting` / a sampler in configs/*.yaml to try to lift rare genres.
- **Concern level:** low-medium

## Final summary
- **Summary:** Milestones 1-6 and 8-10 done; M7 (AST) skipped by user decision. CNN test macro-F1 0.460, MERT 0.493.
- **Still running:** nothing.
- **Needs your attention:** rare-genre failure (above); decide whether to try class-imbalance fixes before adding AST; try the demo in a browser (`python app/demo.py`), including the microphone.
- **Next steps:** (1) AST on days 3-4 via the registry (see README "Adding AST"); (2) several seeds + bootstrap CIs for a fair CNN-vs-MERT comparison; (3) `git init` if you want version control (not a repo yet; nothing committed or pushed).

## [2026-11-16 15:20] Demo: microphone recordings failed (no ffmpeg on this machine)
- **Situation:** User reported an error when submitting a microphone recording. Gradio's default `format="wav"` converts any non-WAV upload server-side with ffmpeg/ffprobe, which is not installed here. Reproduced: a WebM/Opus file fails with the old setting while WAV passes. I did not see the user's exact message, so this is the most likely cause, confirmed by reproduction, not proven for their case.
- **Decision / action taken:** `gr.Audio(format=None)` (no server-side conversion); new `audio.decode_any()` tries libsndfile, then falls back to the ffmpeg binary bundled in the pip package `imageio-ffmpeg` (installed in the venv, added to requirements.txt); errors now print a traceback and append to logs/demo_errors.log, and the page shows the exception type and message. Training/preprocessing `decode()` is untouched, so caches are unchanged. Tested through the real Gradio server: .webm, .ogg, .wav, .m4a, .mp3 all OK; 48 kHz mono/stereo/short/silent/long/streaming-header WAVs OK.
- **How to change it:** app/demo.py (`format=`), src/audio.py (`decode_any`).
- **Concern level:** low. Live (streaming) microphone mode was offered but not requested yet.

## [2026-11-16 16:00] Demo: terminal flooded with "Too little data for declared Content-Length" on recording
- **Situation:** User reported many `h11 LocalProtocolError: Too little data for declared Content-Length` tracebacks when recording with the microphone. Reproduced with real Chrome (headless, fake microphone via Playwright): 5 server errors in 8 recordings, while the browser itself saw nothing wrong (recordings and predictions still worked, so this was log noise plus a possible failed playback fetch).
- **Cause (verified):** Gradio saves an upload at <cache>/<sha256>/<name>. The recorder uploads the same recording twice; `shutil.move` onto the existing file falls back (Windows) to copy-over-existing, which truncates the file while the first request is still serving it, so the declared length exceeds the bytes available.
- **Decision / action taken:** `patch_gradio_upload_race()` in app/demo.py (called from make_ui) replaces `gradio.routes.move_uploaded_files_to_cache` with a version that skips the move when the destination (same content hash = same bytes) already exists. Result: 0 server errors in 12 recordings (baseline reproduced again with 5 in 8). Full demo flow (record, stop, submit, model prediction) verified in headless Chrome.
- **Also found (not fixed, not hit by the demo):** Gradio's `RangedFileResponse` hangs on a Range request that extends past the end of the file (`bytes=0-N`, N > size) and returns HTTP 500 for multi-range requests. Upstream Gradio bugs; the demo's playback does not trigger them.
- **Environment note:** library versions are very new (fastapi 0.163, starlette 0.52, gradio 5.50). Installed `playwright` into the venv for testing only (not in requirements.txt; uses the system Chrome).
- **Limits of verification:** tested with a fake microphone in headless Chrome, not with the user's browser and a real microphone.
- **How to change it:** app/demo.py `patch_gradio_upload_race`.
- **Concern level:** low

## [2026-11-16] Microphone confirmed working by the user
- **Situation:** After allowing the microphone for http://127.0.0.1:7860 in Chrome, recording works on the user's real laptop (the earlier "No microphone found" was a missing browser permission, not a code bug).
- **Decision / action taken:** Closes the "microphone untested" concern. The demo hint text explains the permission step.
- **Concern level:** low

## [2026-11-16] Uploaded the project to GitHub (private)
- **Situation:** User asked to commit to GitHub but had not answered the visibility / email / name / license questions.
- **Decision / action taken:** Created a **private** repository `OpenMAP` under the logged-in `gh` account, pushed branch `main`. Commit identity is repo-local (not global): author name = GitHub login, email = the GitHub noreply address (`ID+login@users.noreply.github.com`), so the user's real email is not published. No LICENSE file. Only source, configs, scripts, README, reports (results.md + 4 small PNG figures), requirements and the decision/progress logs are tracked (~160 KB + ~0.7 MB figures). `.gitignore` excludes `.venv/`, `data/`, `outputs/` (checkpoints, caches, logs), `logs/`. Copied the confusion/UMAP PNGs into `reports/figures/` and fixed their links in `reports/results.md`, because `outputs/` is not tracked.
- **Why:** Private + noreply + no license are the most conservative, easily reversible defaults.
- **How to change it:** Make public: `gh repo edit tanav-bollam/OpenMAP --visibility public --accept-visibility-change-consequences`. Rename: `gh repo rename`. Add a license: add a LICENSE file and commit. Change commit email for future commits: `git config user.email ...` in the repo.
- **Not uploaded:** model checkpoints (`outputs/*/best.pt`), caches, FMA audio/metadata. The MERT head checkpoint is small (~1 MB) and could be added if wanted.
- **Concern level:** low

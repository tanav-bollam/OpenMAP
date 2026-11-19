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

## [2026-11-17] Item 1 done: imbalance sweep + uncertainty (no fix helped)
- **Situation:** Todo item 1: try imbalance fixes, run multiple seeds, add confidence intervals.
- **Decision / action taken:** Added `class_weighting: inv`, `train.sampler` (none|sqrt|inv), a single-file embedding cache (data/embeddings_all*.npy; delete both if embeddings are re-extracted), `src/sweep.py` and `src/analyze_sweep.py`. Ran 5 MERT-head variants x 5 seeds (25 runs, 0 failures, ~3-4 min each) and post-hoc prior correction (tau 0..1) for MERT and the CNN. Selection used VALIDATION only; the test split was read once for the five original-recipe seeds (stage `final`).
- **Result:** The original recipe (loss weights 1/sqrt(freq), no correction) won on validation (0.542 ± 0.013); no variant or tau beat it. Test, original recipe, 5 seeds: macro-F1 0.496 ± 0.004, acc 0.683 ± 0.004. CNN (1 seed) 0.460, 95% CI [0.435, 0.482]. Paired bootstrap MERT − CNN macro-F1: +0.035, 95% CI [0.013, 0.057], 100% of 2,000 resamples > 0. Rare genres unchanged (Blues, Easy Listening 0.00; Soul-RnB 0.18 MERT vs 0.00 CNN).
- **Corrected earlier claims:** results.md/README said the gap "may be noise" and "one seed, no CI"; updated. Remaining caveat: CNN seed variance is unmeasured (one seed).
- **Concerns:** (1) 25 configs compared on validation, so the winning validation score is optimistic (val 0.542 vs test 0.496). (2) The CI covers test-track sampling only, not CNN training variance. (3) Did not rerun the CNN with a fix, because no fix helped. Optional: 2-4 more CNN seeds (~75 min each) to close caveat 2. (4) Files changed but NOT committed or pushed.
- **How to change it:** `python src/sweep.py --variants ... --seeds ...`; `python src/analyze_sweep.py --stage select|final`.
- **Concern level:** low

## [2026-11-17] Housekeeping (T2) done conservatively
- **Situation:** User asked for housekeeping (delete quick-test runs and debug logs, optionally the FMA zips).
- **Decision / action taken:** Deleted `outputs/cnn_quick`, `outputs/mert_quick` and my debug/test logs (`logs/browser*.log`, `crawl*.log`, `demo*.log`, `cnn_quick.log`, `overnight_attempt1.log`), about 0.5 MB plus 21 MB of outputs. Kept all training/sweep/analysis logs and every result. **Did NOT delete** `data/raw/zips/*` (~30 GB), `data/raw/fma_*` or any cache.
- **Why:** The autonomous-run safety rules say never delete raw data while the user is away, and the zips are the only local copy of the verified downloads (re-downloadable, but slow). Everything I deleted is regenerable in minutes.
- **How to change it:** To free ~30 GB yourself once you are sure the extracted audio is complete: `rm data/raw/zips/fma_small.zip data/raw/zips/fma_medium.zip data/raw/zips/fma_metadata.zip`.
- **Concern level:** low

## [2026-11-17 03:35] Plan and design decisions for CNN seeds, error-analysis tests and AST
- **Situation:** User asked for 2-4 more CNN seeds, then housekeeping, error-analysis tests and AST, run autonomously.
- **Decision / action taken:**
  - **CNN seeds:** 3 extra seeds (1, 2, 3) with the unchanged recipe via `scripts/run_cnn_seeds.sh` (log `logs/cnn_seeds.log`, ~80 min each), each evaluated on validation and test; with the original seed 42 that makes 4 seeds. Pre-flight was a 3-epoch quick run on fma_small (passed: val macro-F1 0.443).
  - **Chain 2** (`scripts/run_after_cnn.sh`, log `logs/after_cnn.log`) starts when the seeds finish: re-run `analyze_sweep.py --stage final` (now uses all CNN seeds), re-run `error_analysis.py`, then AST.
  - **Error-analysis tests** (`src/error_analysis.py` -> `reports/error_analysis_tests.md`): 5 tests (scarcity, label ambiguity via multi-tag, kNN overlap, pairwise separability, shared errors). **H2 (multi-tag) is not testable**: all 25,000 FMA-medium tracks have exactly one top-level genre by construction (verified; 54,745 of 106,574 tracks in the full FMA have several), so I report it as 'not testable' instead of 'not supported'.
  - **AST** (`src/models/ast_model.py`, `configs/ast.yaml`): HF checkpoint `MIT/ast-finetuned-audioset-10-10-0.4593`; GPU feature extraction verified identical to `ASTFeatureExtractor` (max abs diff 0.0); non-reentrant gradient checkpointing (needed so trainable layers get gradients when early layers are frozen; verified on CPU); embeddings + first 8 of 12 layers frozen (33% of 86M parameters trainable); head lr 1e-3, backbone lr 1e-5; same loss recipe as the other models. **This is a reduced fine-tune, not the full 10-15 epoch recipe from the brief.** `scripts/time_ast.py` measures the real step time and picks `data.train_fraction` (stratified) so that 3 epochs + validation fit an `AST_BUDGET_HOURS` budget (default 3 h, floor 10%).
  - **Demo:** added a `wave` branch in `app/demo.py` so the AST run can be served.
- **Why:** Memory is tight (15 GB), so nothing GPU-heavy runs alongside the CNN; everything is chained to run in sequence, detached, and each step is resumable or re-runnable.
- **How to change it:** `AST_BUDGET_HOURS=6 bash scripts/run_after_cnn.sh` (or edit `configs/ast.yaml`: `freeze_layers`, `epochs`, `train_fraction`); more CNN seeds: edit the `for seed in ...` line in `scripts/run_cnn_seeds.sh`.
- **Not done on purpose:** no `git commit`/`push` (safety rules for unattended runs); raw zips kept.
- **Concern level:** medium (AST is untested on GPU until the CNN finishes; if it fails the chain logs it and stops AST only)

## [2026-11-17 08:10] T1 (CNN seeds) and T3 (error-analysis tests) finished
- **Situation:** 3 extra CNN seeds finished at 08:09 (about 95 min each, slower than the first run because of laptop heat, GPU ~84 C / 1770 MHz).
- **Result:** CNN test macro-F1 per seed 0.460 (seed 42), 0.461, 0.470, 0.468 -> 0.465 ± 0.005 (accuracy 0.673 ± 0.002). With 5 MERT seeds (0.496 ± 0.004) the paired-bootstrap MERT − CNN gap is +0.031 (95% CI 0.010 to 0.050; MERT ahead in 100% of resamples). The earlier open caveat (CNN seed variance unmeasured) is closed: seed spread is ~6x smaller than the gap.
- **Error-analysis tests** (reports/error_analysis_tests.md): H1 scarcity supported (rho 0.58/0.59, p<0.02); H2 label ambiguity NOT TESTABLE (all FMA-medium tracks have exactly one top-level genre by construction); H3 rare genres have 0-1% same-genre neighbours (consistent with dispersion/overlap); H4 confused pairs poorly separable even with plenty of data (Instrumental vs Folk AUC 0.75 vs 0.998 for control pairs); H5 models share 79-81% of errors and agree on the wrong label 70% of the time (13% by chance). Interpretation kept careful in results.md: these tests cannot separate noisy labels from real acoustic overlap.
- **Docs updated:** reports/results.md (caveats, tests table), README, reports/imbalance_and_uncertainty.md (regenerated with 4 CNN seeds).
- **Concern level:** low

## [2026-11-17 08:12] AST is much faster than estimated: queued a full fine-tune as a second run
- **Situation:** `scripts/time_ast.py` measured 0.31 s per batch of 8 with 8 layers frozen + gradient checkpointing + bf16 (26 samples/s, peak 1.2 GB GPU), i.e. ~13 min per full epoch over 19,904 training tracks (my earlier estimate was 1-2 h per epoch). The time budget therefore chose train_fraction = 1.0 and the whole reduced run is estimated at ~0.8 h.
- **Decision / action taken:** Chain 2 continues unchanged and trains the reduced AST (`outputs/ast`: first 8 layers frozen, 3 epochs, 100% of the training data). Added `scripts/run_ast_full.sh`, which waits for chain 2 and then times a FULL fine-tune (`AST_FREEZE=0`), picks 3-8 epochs that fit `AST_BUDGET_HOURS` (default 3), and trains `outputs/ast_full`. Having both gives a free ablation of freezing vs full fine-tuning, and the full run is closer to the brief's recipe (the brief asked for 10-15 epochs; the budget will allow fewer).
- **Why:** The measurements show the reduced setup does not need to be so reduced; a better-quality AST is more informative for the CNN/MERT/AST comparison, and the cost is only GPU time.
- **How to change it:** `AST_BUDGET_HOURS=6 bash scripts/run_ast_full.sh` for more epochs (max 8 in the script); `configs/ast.yaml` for learning rates, `freeze_layers`.
- **Concern level:** low (if the full run OOMs, `train.py` falls back to batch 4 automatically; failures are logged in logs/ast_full_chain.log and do not affect other results)

## [2026-11-17 08:40] AST timing corrected for heat throttling
- **Situation:** Reduced AST run epoch 1 took 1528 s (val acc 0.722, macro-F1 0.549) vs 926 s predicted by `time_ast.py` (cool-GPU step time). Real epochs are 1.65x slower (GPU 84-88 C, clock ~1400-1760 MHz of 2100).
- **Decision / action taken:** `scripts/time_ast.py` now multiplies timings by `AST_SLOWDOWN` (default 1.65) before choosing the number of epochs for the full fine-tune. The reduced run is unaffected (it finishes ~09:30).
- **How to change it:** `AST_SLOWDOWN=1.0` or `AST_BUDGET_HOURS=...` environment variables when launching `scripts/run_ast_full.sh`.
- **Concern level:** low. Note the first AST epoch already beats the MERT head on validation macro-F1 (0.549 vs 0.542); the test numbers will decide the comparison.

## [2026-11-17 09:40] AST (reduced fine-tune) result; demo verified with AST
- **Result (outputs/ast, seed 42, 8 layers frozen, 3 epochs, 100% of the training data, 1528 s/epoch):** best val macro-F1 0.576 (epoch 3, val acc 0.729). **Test: acc 0.708, top-3 0.862, macro-F1 0.538, weighted-F1 0.701.** Paired bootstrap on test macro-F1: AST − MERT = +0.042 (95% CI 0.013 to 0.073), AST − CNN = +0.073 (0.042 to 0.102). First non-zero Blues F1 (0.18); Country 0.60 (MERT 0.34); Easy Listening still 0.00.
- **Caveats:** AST is ONE run (seed variance unmeasured; MERT/CNN seed std was ~0.005, fine-tuned transformers can vary more); the test intervals cover test-track sampling only; only 3 epochs (the brief asked for 10-15) and lower layers frozen, so this probably understates what AST can do.
- **Demo:** `python app/demo.py` now auto-picks the run with the best test macro-F1, which is now `ast`. Verified through the Predictor: correct top-3 on a Hip-Hop track, a WebM recording, a 0.5 s clip, silence (finite, sums to 1) and a 45 s clip. The full web UI was not re-tested in a browser with AST.
- **Next:** full fine-tune `ast_full` (all layers, 3 epochs, est. ~47 min/epoch; ETA ~12:10), then re-run `analyze_sweep.py --stage final` and update the docs.
- **Concern level:** low

## [2026-11-17 11:55] T4 (AST) finished: full fine-tune result and final comparison
- **Result:** full fine-tune (`outputs/ast_full`, all layers trainable, 3 epochs, ~40 min/epoch): best val macro-F1 0.573, **test acc 0.702, top-3 0.856, macro-F1 0.531**. Reduced run (`outputs/ast`): test acc 0.708, macro-F1 0.538. Difference full − reduced −0.007 (95% CI −0.023 to 0.009): no measurable gain from unfreezing in 3 epochs.
- **Final ranking (test macro-F1):** AST 0.531-0.538 > MERT 0.496 ± 0.004 (5 seeds) > CNN 0.465 ± 0.005 (4 seeds). Paired-bootstrap gaps: AST − MERT +0.035 to +0.042 (CIs exclude 0), MERT − CNN +0.031.
- **Docs updated:** README (three-model table, AST section, lessons), reports/results.md (new AST section, AST confusion figure in reports/figures), reports/imbalance_and_uncertainty.md (regenerated). `evaluate.py outputs/... ` without `--no-report` would OVERWRITE reports/results.md (it regenerates it from scratch) - do not run it without backing up the manual sections.
- **Open caveats:** (1) AST is a single run per setting; seed variance unmeasured. (2) 3 epochs, not the brief's 10-15. (3) Easy Listening is 0.00 for all models (6 test tracks). (4) The demo web UI with AST was not re-tested in a browser (the Predictor path was). (5) The GPU is heat-throttled (84-88 C), so timings in the docs are for this laptop.
- **Concern level:** low

## Final summary (autonomous run 2)
- **Summary:** All four tasks finished. CNN seeds (4 total: 0.465 ± 0.005 macro-F1), housekeeping (test artifacts and debug logs removed; raw zips kept), error-analysis tests (H1, H3, H4, H5 supported/consistent; H2 not testable on single-genre FMA-medium), AST (best model: test macro-F1 0.538, acc 0.708).
- **Still running:** nothing. All chains finished (logs/cnn_seeds.log, logs/after_cnn.log, logs/ast_full_chain.log).
- **Needs your attention:** (1) Nothing is committed or pushed: `git status` shows many modified and new files (new scripts, AST model, reports, docs). Say the word and I will commit and push to the private repo. (2) Output directory sizes: `outputs/` now holds CNN (x4), MERT sweep (25 runs), AST (x2) checkpoints (1.1 GB in total). (3) Free ~30 GB by deleting `data/raw/zips/*` if you no longer need the zips.
- **Next steps:** run more seeds of AST (~75 min per 3-epoch run), try 10-15 epochs for AST as in the brief, or compare against MERT fine-tuning; decide whether to merge/drop the rare genres (Blues, Easy Listening, Soul-RnB), the one issue every model shares.

## [2026-11-17] Priority AST run prepared but not started (user will run it tonight)
- **Decision / action taken:** Wrote `scripts/run_ast_seeds.sh` and `run_ast_seeds.bat` (2 extra seeds of the reduced AST recipe, one 10-epoch run with early-stop patience 4, then the final analysis). Updated `src/analyze_sweep.py` so that AST seeds `outputs/ast_s*` are pooled with `outputs/ast` (seed mean ± std) and `outputs/ast_e10` is its own row; dry-run on the current results passes. Estimated ~7 h total (~1h25 per 3-epoch run, ~4h20 for 10 epochs). Saved as a memory item.
- **Why:** The user asked to keep this as the priority item. Answers whether AST's lead over MERT (+0.035 to +0.042 macro-F1, single run) is real, and what a longer run gives.
- **Concern level:** low. These changes are NOT committed or pushed yet. A 10-epoch run may stop early (patience 4); the .bat window must stay open (or use the nohup command, which survives a closed terminal).

## [2026-11-17] Demo with AST confirmed working by the user in a browser
- **Situation:** The user ran `run_demo.bat`; the terminal printed `serving ast (ast)` and the page served normally, and they report recording/uploading worked.
- **Decision / action taken:** Closes the open caveat "demo web UI with AST not re-tested in a browser". The `You should probably TRAIN this model...` line printed at startup is a harmless transformers notice about the re-initialised 16-class head before the trained weights are loaded.
- **Concern level:** low

## [2026-11-17] Rare genres: decided to leave as a documented limitation
- **Situation:** Open question whether to merge or drop Blues, Easy Listening and Soul-RnB (near-zero F1 for most models).
- **Decision / action taken:** User chose to leave it. No change to the 16-genre task. The limitation stays documented in README and reports/results.md (training counts 13-94 vs 5,681 for Rock; loss weighting, sampling and prior correction did not help; Easy Listening is 0.00 for all models).
- **Why:** Merging/dropping would change the task and make results incomparable with the current ones; it is not required by the brief.
- **How to change it:** Revisit later by merging classes in `src/prepare_metadata.py` and retraining, or by using FMA sub-genre labels.
- **Concern level:** low

## [2026-11-17 21:56] Priority AST run STARTED (2 extra seeds + 10-epoch run)
- **Situation:** User asked to start the planned AST run now. GPU idle (0 MiB), demo closed, 3.5 GB RAM free.
- **Decision / action taken:** Launched `scripts/run_ast_seeds.sh` detached (log `logs/ast_seeds.log`, per-run logs `logs/ast_s1_train.log`, `ast_s2_train.log`, `ast_e10_train.log`). Order: `ast_s1` (seed 1) -> `ast_s2` (seed 2) -> `ast_e10` (10 epochs, early-stop patience 4) -> final analysis. All three use the reduced recipe (8 layers frozen, batch 8, all training data). The recipe is identical to the already-tested reduced run apart from seed/epochs, and `train.py` runs its own one-batch smoke test first, so no separate quick test was run.
- **ETA (estimates, +-15%):** ast_s1 ~23:20, ast_s2 ~00:45, ast_e10 ~05:00, analysis ~05:15 (2026-11-18). Finished marker: `logs/ast_seeds_done`.
- **How to stop it:** kill the `bash scripts/run_ast_seeds.sh` process and its python child (Task Manager or `taskkill`); rerunning the script resumes and skips finished runs.
- **Concern level:** low (laptop must stay plugged in/awake; heat throttling may lengthen it)

## 2026-11-18: AST seeds and 10-epoch run results
- 3 AST seeds (8 frozen, 3 epochs): test macro-F1 0.538/0.531/0.531, spread 0.004 like CNN/MERT, so the AST lead over MERT (+0.038, 95% CI 0.012 to 0.064) is real.
- 10-epoch run: best val macro-F1 0.559 at epoch 6, then overfit; early stop after epoch 10; test macro-F1 0.510 (0.024 below the 3-epoch recipe, CI -0.046 to -0.002). Concern: single run, so read as "no evidence more epochs help". Decision: keep the 3-epoch recipe as the headline AST result.
- Concern: the user changed the Windows power plan mid-run (GPU at 817 MHz), adding about 30% to epoch times; no effect on results.

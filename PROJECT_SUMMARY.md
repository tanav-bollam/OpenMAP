# Project summary: music genre recognition on FMA-medium

This is the one-file story of the project, from the first plan to the finished demo. For the details behind
any point, see the files listed at the end.

## 1. The goal

Build and compare three ways of recognising the genre of a music clip, using the FMA-medium dataset
(25,000 tracks, 16 genres, 30-second clips), as described in `PROJECT_BRIEF.md`:

- **Model A, CNN from scratch.** A small convolutional network (about 5 million parameters) trained only on
  our data, reading mel-spectrograms.
- **Model B, AST (Audio Spectrogram Transformer).** A large pretrained transformer (86 million parameters),
  fine-tuned for genres.
- **Model C, MERT + head.** A pretrained music model (MERT-v1-95M) kept frozen, with a small trained
  classifier on top of its embeddings.

Plus a web demo, an evaluation report and a README.

## 2. The machine, and how the plan was adapted

The brief assumed a big GPU. The real machine is a laptop with an RTX 3060 (6 GB), 15 GB RAM and Windows.
The audit at the start concluded that AST was risky on this hardware, so the plan became:

- Start with CNN vs MERT (the two-day goal), and build the code so AST could be added later. AST was added
  afterwards and did fit, with a reduced recipe.
- Decode each MP3 once and cache two versions: a 22.05 kHz log-mel spectrogram (for the CNN) and a 24 kHz
  waveform (for MERT and, after resampling, AST).
- Always use mixed precision, run a smoke test and a quick test before every full run, and train the CNN
  with batch 32, at most 30 epochs, stopping early on validation macro-F1.
- Log every decision and concern in `DECISIONS_AND_CONCERNS.md` and run unattended where possible.

## 3. What was built

| Stage | What happened |
|---|---|
| Data | Official artist-separated split. 24,980 of 25,000 tracks cached; 20 corrupt or missing tracks were logged and skipped. Test set: 2,572 tracks. |
| CNN | Five convolution blocks with average and max pooling. Trained on 10-second windows. |
| MERT | Embeddings extracted once (13 layers x 768 values per track), then a small head with learnable layer weights was trained. Needed `transformers<5` and a pinned model revision. |
| AST | Added later through the pluggable model registry. 16 kHz audio, 10.24 s crops, first 8 of 12 layers frozen, 3 epochs, class-weighted loss with label smoothing. |
| Evaluation | One script for all models. Track-level prediction averages the probabilities over windows. The test set is read once per run; all choices are made on validation data. |
| Demo | A web page with upload and microphone recording (see section 6). |

## 4. Results (test set, 2,572 tracks)

| Model | Runs | Top-1 accuracy | Macro-F1 |
|---|---|---|---|
| CNN from scratch | 4 seeds | 0.673 +/- 0.002 | 0.465 +/- 0.005 |
| MERT + head | 5 seeds | 0.683 +/- 0.004 | 0.496 +/- 0.004 |
| **AST (8 layers frozen, 3 epochs)** | 3 seeds | **0.705 +/- 0.003** | **0.533 +/- 0.004** |
| AST, full fine-tune | 1 run | 0.702 | 0.531 |
| AST, 10 epochs | 1 run | 0.682 | 0.510 |

Accuracy is the share of tracks given the right genre. Macro-F1 averages the F1 score of all 16 genres
equally, so rare genres count as much as Rock. It is the stricter measure and the main one in the brief.
AST's top-3 accuracy is about 0.86.

What the numbers show:

- **The ranking is real.** Paired bootstrap over the test tracks: MERT beats the CNN by +0.031 macro-F1
  (95% CI 0.010 to 0.050), and AST beats MERT by +0.038 (0.012 to 0.064). The three AST seeds agree
  (0.538, 0.531, 0.531), so this is not luck of one seed.
- **Pretraining pays off**, and a from-scratch CNN gets surprisingly close to frozen pretrained features.
- **More epochs did not help AST.** The 10-epoch run peaked on validation at epoch 6 and then overfit
  (train accuracy 0.86, validation 0.71), so it scored 0.024 lower than the 3-epoch recipe.
- **Unfreezing all layers made no measurable difference** to freezing 8 (-0.003, CI -0.017 to 0.012), and was
  about 1.6x slower.

## 5. Side studies

- **Class imbalance.** The genres are very uneven (Rock 5,681 training tracks, Easy Listening 13). I tested
  other loss weightings, class-balanced sampling and post-hoc prior correction across 5 seeds each. None
  beat the original recipe on validation, so it was kept. Details: `reports/imbalance_and_uncertainty.md`.
- **Rare genres.** Blues, Easy Listening and Soul-RnB stay near zero F1 for every model (AST gets 0.16 on
  Blues). You decided to leave merging or dropping them as a documented limitation.
- **Error analysis.** Five explanations for the models' mistakes were turned into tests. Four were
  supported or consistent with the data. One (genre ambiguity through multiple tags) could not be tested
  because FMA-medium gives each track exactly one top-level genre. Details: `reports/error_analysis_tests.md`.

## 6. The demo

The Music Genre Classifier web page lets you drag in an audio file or record from the microphone and shows
the top three genres, the confidence for each, and the spectrogram. It started as a Gradio app and went
through several fixes: microphone recordings failing without ffmpeg, a Windows upload bug in Gradio, a
"No microphone found" message caused by browser permissions, and a startup that looked frozen. In the end it
was replaced by a custom page (`app/web/index.html`) served by a small FastAPI server (`app/demo.py`).
The page opens immediately and the model loads in the background. The old Gradio version is kept as
`app/demo_gradio.py`.

To run it: double-click `run_demo.bat`, or `.\.venv\Scripts\python app\demo.py`, then open
`http://127.0.0.1:7860`.

## 7. Problems met along the way

- MERT returned no hidden states under `transformers` 5, so the code is pinned to version 4.
- Windows spawn-style DataLoader workers each held about 0.6 GB, which slowed training until the number of
  validation workers was capped.
- Windows text encodings and line endings caused crashes and noisy diffs; everything is now UTF-8 with LF.
- A short timing test on a cool GPU under-estimated real AST epoch time by about 1.65x because the laptop
  throttled when hot. The Windows "Silent" power plan also cut the GPU clock by half; "Performance" fixed it.
- Background processes were occasionally killed under memory pressure, so long jobs ran as detached chains
  that can be resumed.

## 8. Timeline

| When | Milestone |
|---|---|
| 2026-11-16 | Audit, plan, data download and caching, CNN and MERT trained, first report, demo and README |
| 2026-11-17 | Demo fixes, GitHub upload, imbalance sweep and bootstrap intervals, extra CNN seeds, error-analysis tests, AST implemented and evaluated |
| 2026-11-17 to 18 (overnight) | Two more AST seeds and the 10-epoch run, about 7 hours |
| 2026-11-18 | Final analysis, docs updated, new web page, final push |

Git history: `4d58abb` first version, `886a8da` AST, extra seeds and studies, `5a8e7cc` AST seeds and
10-epoch results, `6d65051` the new web demo.

## 9. Limitations

- Rare genres are essentially not learned, so macro-F1 is held back by about three genres.
- AST was trained for 3 epochs, not the brief's 10 to 15. The 10-epoch run showed more epochs did not help.
- The full fine-tune and 10-epoch AST rows are single runs.
- Validation scores are optimistic relative to test, as for the other models.
- The microphone path of the new web page was not tested with a real microphone in the build session.
- The labels themselves are noisy: FMA genres are assigned by artists and overlap heavily.

## 10. Where to find things

| File | Contents |
|---|---|
| `README.md` | How to run, headline results, file layout |
| `reports/results.md` | Main evaluation report with per-genre tables and confusion matrices |
| `reports/imbalance_and_uncertainty.md` | Imbalance sweep, seed spreads, confidence intervals |
| `reports/error_analysis_tests.md` | Tests of the error explanations |
| `DECISIONS_AND_CONCERNS.md` | Every decision and concern, in order |
| `PROGRESS.md` | Milestone checklist |
| `PROJECT_BRIEF.md` | The original spec |
| `configs/*.yaml`, `src/`, `scripts/`, `app/` | Settings, code, run scripts and the demo |

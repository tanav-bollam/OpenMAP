# Music genre recognition (FMA-medium): CNN vs MERT

Predict the genre of a clip (up to 30 s) with top-3 probabilities. Two models are compared on the
official, artist-separated FMA-medium test split:

- **A. CNN from scratch** on 22.05 kHz log-mel spectrograms (4.98M parameters).
- **C. Frozen MERT-v1-95M embeddings** + a learnable layer-weighted sum and a small MLP head.
- **B. AST** is intentionally not implemented yet (planned for later); the code is set up to add it
  (see "Adding AST").

## Results (official test split, 2,572 tracks, track level)

| Model | Top-1 acc | Top-3 acc | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| A. CNN (from scratch) | 0.671 | 0.845 | 0.460 | 0.651 |
| C. MERT + head | **0.682** | 0.846 | **0.493** | **0.680** |

Full per-genre table, confusion matrices and error analysis: [reports/results.md](reports/results.md).
Per-run files are in `outputs/<run>/` (`classification_report_test.txt`, `confusion_test.png`, `metrics.csv`).

Honest reading of the numbers:
- MERT is better, but only modestly (one seed, no confidence intervals; rare genres have 6-42 test tracks).
- Blues, Easy Listening and Soul-RnB are essentially not learned by either model (F1 0.00-0.08).
  Training counts are 58, 13 and 94 tracks against 5,681 for Rock.
- Validation macro-F1 is optimistic for MERT (0.544 val vs 0.493 test) because the best epoch was
  picked on a noisy metric. Quote the test numbers.

## Reproduce

Requirements: Python 3.12, an NVIDIA GPU (developed on a 6 GB RTX 3060 laptop), ~100 GB free disk
(measured: 7.8 GB of mels, 34 GB of 24 kHz waveforms, 1 GB of MERT embeddings, and 62 GB of raw
zips plus unzipped audio, about 105 GB in total; delete the zips afterwards to save ~30 GB).
No ffmpeg is needed (libsndfile decodes MP3).

```bash
python -m venv .venv && .venv/Scripts/activate            # Windows; use bin/activate elsewhere
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130   # match your driver
pip install -r requirements.txt

# data: download + verify SHA1 (see https://github.com/mdeff/fma), unzip into data/raw/
#   data/raw/fma_metadata/  data/raw/fma_small/  data/raw/fma_medium/

python src/prepare_metadata.py --subset small
python src/prepare_metadata.py --subset medium
python src/preprocess_mels.py --subset small              # one MP3 decode -> mel cache + 24 kHz waveform cache
python src/preprocess_mels.py --subset medium             # ~45 min; logs bad files to data/bad_tracks.txt

# pre-flight checks (also run automatically before each full run)
python src/train.py --config configs/cnn.yaml --smoke     # one batch forward/backward
python src/train.py --config configs/cnn.yaml --quick     # 3 epochs on fma_small

python src/train.py --config configs/cnn.yaml             # Model A, ~75 min
python src/extract_mert.py --config configs/mert.yaml     # ~30 min, resumable
python src/train.py --config configs/mert.yaml            # Model C, ~1 min
python src/evaluate.py outputs/cnn outputs/mert --umap    # writes reports/results.md

python app/demo.py                                        # Gradio demo (best evaluated model)
# Windows without activating the venv: .\.venv\Scripts\python app\demo.py   (or double-click run_demo.bat)
```

`bash scripts/run_overnight.sh` chains everything after the data is unzipped. Every step is resumable.

## Design notes

- **One decode pass.** Each MP3 is decoded once; the script writes a 22.05 kHz log-mel
  (`data/mels`, float16, 128x1292) and a 24 kHz waveform (`data/wav24k`, float16). MERT reads the waveform;
  `audio.resample_on_load` / `datasets.WaveDataset` turn it into any other rate (e.g. 16 kHz for AST).
- **Splits.** FMA's official `training/validation/test` split, filtered with `subset in {small, medium}`.
  (Comparing the subset column as a string with `<= 'medium'` would wrongly include `large`.)
  Mel normalization statistics come from the training split only.
- **Evaluation.** Track level: five 10 s windows (5 s hop) per clip, softmax probabilities averaged.
- **Imbalance.** Cross-entropy weights proportional to 1/sqrt(class frequency) (not a sampler), label smoothing 0.1.
  Macro-F1 is the selection metric (early stopping and best checkpoint).
- **CNN training.** AdamW, OneCycle, bf16, batch 32, 30 epochs, SpecAugment, +-6 dB gain, mixup 0.2.
  Reported train accuracy is measured against pre-mixup labels, so it understates real training accuracy.
- **MERT.** fp16, 5 s chunks, mean over time and chunks, 13 layers x 768. The code is pinned to
  `transformers<5` and a specific MERT code revision: under transformers 5.x the remote code returned no hidden states.

## Adding AST (planned)

1. Add `src/models/ast_model.py` with `@register("ast")`, import it in `src/models/__init__.py`.
2. The model takes a waveform batch (`data.kind: wave`, 16 kHz crops via `WaveDataset`) and applies
   `ASTFeatureExtractor` inside `forward()`; give it a `param_groups(cfg)` for backbone/head learning rates.
3. Fill in `configs/ast.yaml` (placeholder), then `python src/train.py --config configs/ast.yaml`.
   `evaluate.py` and the demo are model-agnostic through `build_model` and the checkpoint config.
   (The demo needs a small branch for `data.kind: wave`.)

AST fine-tuning is heavy for a 6 GB GPU (use gradient checkpointing, small batches, few epochs).

## Lessons learned

- A from-scratch CNN gets surprisingly close to frozen pretrained embeddings on FMA genres; label noise
  and class imbalance limit both, more than model capacity does.
- Rare classes dominate macro-F1 and are the least reliable part of the evaluation.
- On Windows (spawn), idle persistent DataLoader workers each hold ~0.6 GB; on a 15 GB machine this
  slowed training until validation workers were capped at 2 and made non-persistent.
- 20 of 25,000 FMA-medium tracks are missing or corrupt; the pipeline logs and skips them.

## Layout

```
configs/  cnn.yaml mert.yaml ast.yaml(placeholder)     src/  data, models, train, evaluate, extract_mert
app/demo.py  Gradio demo                               reports/results.md  comparison + error analysis
scripts/run_overnight.sh  end-to-end chain             data/ outputs/ logs/  (gitignored)
```

Data: Free Music Archive (Defferrard et al., ISMIR 2017). Track licenses vary (Creative Commons);
do not redistribute the audio.

# Project Brief: Music Genre Recognition with Deep Learning

> **For Claude Code:** This is the full specification. Build the project described here, following the milestones in order. Keep the code modular, reproducible (fixed seeds, config files), and runnable from the command line. Ask before deviating from the architecture or data choices below.

---

## 1. Goal

Build a system that takes a short audio clip (up to 30 seconds) and predicts its music genre, with probabilities for the top genres. Compare a CNN trained from scratch against pretrained audio models, and ship a small interactive demo.

**Time budget:** 2 days.
**Hardware:** a single strong NVIDIA GPU, a multi-core CPU, and about 100 GB of free disk space.

### Deliverables
1. A preprocessing pipeline that turns MP3s into cached log-mel spectrograms.
2. Three trained models, compared on the same test set:
   - **A.** CNN trained from scratch on mel-spectrograms
   - **B.** Audio Spectrogram Transformer (AST), fine-tuned
   - **C.** Frozen pretrained music embeddings (MERT) with a small classifier on top
3. An evaluation report: accuracy, macro-F1, per-genre metrics and a confusion matrix.
4. A Gradio demo: upload or record a clip and see the top 3 genres with probabilities.
5. A README with results, the commands to reproduce them, and short notes on what worked.

---

## 2. Data

### Main dataset: FMA (Free Music Archive)
- Paper: Defferrard et al., "FMA: A Dataset For Music Analysis", ISMIR 2017 (arXiv:1612.01840)
- Repo with official loaders and notes: https://github.com/mdeff/fma
- Known issues (errata): https://github.com/mdeff/fma/wiki
- License: Creative Commons (it varies by track); the metadata is free to use.

**Download links (official):**
```
https://os.unil.cloud.switch.ch/fma/fma_metadata.zip   # required (~342 MB)
https://os.unil.cloud.switch.ch/fma/fma_small.zip      # 8,000 clips, 8 balanced genres (~7.2 GB)
https://os.unil.cloud.switch.ch/fma/fma_medium.zip     # 25,000 clips, 16 genres (~22 GB) ← MAIN
```
Check the SHA1 checksums listed in the repo README after downloading. **Do not** download fma_large or fma_full.

**Which subset is used for what:**
- `fma_small` → quick pipeline checks (Milestone 1–2).
- `fma_medium` → all reported results.

**Labels:** use the top-level genre column `('track', 'genre_top')` from `tracks.csv`. The 16 genres in fma_medium are imbalanced (Rock and Electronic dominate); handle this as described in Section 5.

**Splits:** use FMA's official split in `tracks.csv` → `('set', 'split')` with values `training` / `validation` / `test`, filtered by `('set', 'subset') <= 'medium'`. This split keeps each artist on one side only, so test results are not inflated. **Never make a random split.**

**Loading `tracks.csv`:** it has a two-row header. Use the `utils.py` loader from the FMA repo (`utils.load('tracks.csv')`), or `pd.read_csv(path, index_col=0, header=[0, 1])`.

**File paths:** the audio for track ID 2 is at `fma_medium/000/000002.mp3` (the folder is the first 3 digits of the 6-digit zero-padded ID).

**Known bad files:** a handful of MP3s in fma_medium are empty, corrupt, or much shorter than 30 seconds. The preprocessing must catch decode errors, log the skipped IDs to `data/bad_tracks.txt`, and continue.

### Sanity check (optional)
GTZAN (Hugging Face `marsyas/gtzan`, or Kaggle) only as an extra sanity check. Do not report results on it; it has known duplicates and mislabels (Sturm, 2013).

---

## 3. Project structure

```
music-genre/
├── README.md
├── requirements.txt
├── configs/
│   ├── cnn.yaml
│   ├── ast.yaml
│   └── mert.yaml
├── data/                     # gitignored
│   ├── raw/                  # unzipped fma_metadata, fma_small, fma_medium
│   ├── mels/                 # cached .npy spectrograms
│   ├── embeddings/           # cached MERT embeddings
│   ├── splits.csv            # track_id, genre, label_idx, split
│   └── bad_tracks.txt
├── src/
│   ├── prepare_metadata.py   # builds splits.csv
│   ├── preprocess_mels.py    # MP3 → log-mel .npy, run in parallel
│   ├── extract_mert.py       # MP3 → MERT embeddings
│   ├── datasets.py           # PyTorch Dataset classes + augmentation
│   ├── models/
│   │   ├── cnn.py
│   │   ├── ast_model.py
│   │   └── mert_head.py
│   ├── train.py              # one training script, model chosen by config
│   ├── evaluate.py           # metrics, confusion matrix, report
│   └── utils.py
├── app/
│   └── demo.py               # Gradio demo
├── outputs/                  # checkpoints, logs, plots (gitignored)
└── reports/
    └── results.md
```

---

## 4. Data pipeline

### 4.1 Metadata (`prepare_metadata.py`)
- Load `tracks.csv`; keep rows where the subset is `small` or `medium` (for fma_medium) and `genre_top` is not empty.
- Map the genres to integer labels (sorted alphabetically, saved to `data/genres.json`).
- Write `data/splits.csv` with these columns: `track_id, path, genre, label, split`.
- Print the number of tracks per genre in each split.

### 4.2 Mel-spectrogram cache (`preprocess_mels.py`)
Decoding MP3s is the main bottleneck, so do it **once**:
- Decode with `librosa.load(path, sr=22050, mono=True)` (or torchaudio with an ffmpeg backend).
- Pad or trim to exactly 30 s (661,500 samples).
- Log-mel: `n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=11025`, then `power_to_db`. That gives an array of shape `(128, 1292)`.
- Save as `float16` `.npy` to `data/mels/{track_id}.npy` (~330 KB each, about 8 GB for fma_medium).
- Run in parallel with `multiprocessing.Pool` (or `joblib`) over all CPU cores, with a `tqdm` progress bar.
- Skip files that already exist, so the script can resume after an interruption.
- Compute the mean and standard deviation over the **training split only**, save them to `data/mel_stats.json`, and use them to normalize the inputs.

Expected run time for fma_medium: roughly 30–60 minutes on a modern multi-core CPU.

### 4.3 Training data loading (`datasets.py`)
- `MelDataset` loads the cached `.npy` files and normalizes them.
- **Training:** take a random 10-second crop (~430 frames).
- **Validation and test:** split the 30-second clip into overlapping 10-second windows, run the model on each, and **average the probabilities** to get one prediction per track.
- **Augmentation (training only):** SpecAugment (2 frequency masks of width ≤ 20, 2 time masks of width ≤ 40), random gain of ±6 dB, and **mixup** (α = 0.2) applied per batch.

---

## 5. Models

### Model A: CNN from scratch (`models/cnn.py`)
A compact 2D CNN on log-mel input `(1, 128, T)`:
- 5 conv blocks: `Conv2d(3×3) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm → ReLU → MaxPool(2×2)`, with channels 32 → 64 → 128 → 256 → 512.
- Global pooling: average + max pooling over frequency and time, concatenated.
- Head: `Dropout(0.5) → Linear(1024, 256) → ReLU → Dropout(0.3) → Linear(256, n_classes)`.
- Around 5M parameters.

**Training:** AdamW (lr 1e-3, weight decay 1e-4), OneCycle or cosine schedule with warmup, batch size 64, 40–60 epochs, mixed precision (bf16/fp16), early stopping on validation macro-F1.

### Model B: Audio Spectrogram Transformer (`models/ast_model.py`)
- Checkpoint: `MIT/ast-finetuned-audioset-10-10-0.4593` (Hugging Face `transformers`; `ASTForAudioClassification` + `ASTFeatureExtractor`).
- **Important:** AST expects **16 kHz** audio and its own feature extractor (128 mel bins, 10-second input). It cannot use the 22.05 kHz mel cache above; build a second cache at 16 kHz with the AST feature extractor (`data/ast_feats/`), or compute features on the fly from resampled audio.
- Replace the classification head with one for the 16 FMA genres (`num_labels=16, ignore_mismatched_sizes=True`).
- Fine-tune: AdamW, lr 1e-5 for the backbone and 1e-3 for the head, batch size 16–32, 10–15 epochs, mixed precision, gradient checkpointing if memory is tight.

### Model C: MERT embeddings + small classifier (`models/mert_head.py`, `extract_mert.py`)
- Checkpoint: `m-a-p/MERT-v1-95M` (requires `trust_remote_code=True`). It expects **24 kHz** mono audio.
- Extract embeddings **once**: run the frozen MERT on each 30-second clip (in 5–10 second chunks if needed for memory), take the hidden states from all layers, average over time, and save an array of shape `(n_layers, 768)` per track to `data/embeddings/`.
- Classifier: a learnable weighted sum over layers → `Linear(768, 256) → ReLU → Dropout(0.3) → Linear(256, n_classes)`.
- Trains in minutes. This is the fastest route to a strong result and the **fallback if Day 2 runs short**.

### Class imbalance (all models)
- Use a class-weighted cross-entropy loss (weights ∝ 1/√frequency) **or** a weighted sampler. Pick one and say which in the README.
- Use label smoothing of 0.1.
- Report **macro-F1** as the main metric (accuracy alone hides poor performance on rare genres).

---

## 6. Training setup (`train.py`)
- One script for all three models, driven by a YAML config: `python src/train.py --config configs/cnn.yaml`.
- Fixed seeds (Python, NumPy, PyTorch); `torch.backends.cudnn.benchmark = True`.
- Mixed precision with `torch.autocast` and gradient scaling for fp16.
- Log training/validation loss, accuracy and macro-F1 every epoch to TensorBoard (or Weights & Biases if available).
- Save the best checkpoint by validation macro-F1 to `outputs/{run_name}/best.pt`, along with the config used.
- `DataLoader`: `num_workers` = number of CPU cores minus 2, `pin_memory=True`, `persistent_workers=True`.

---

## 7. Evaluation (`evaluate.py`)
For each model, on the **official test split**, at the track level (averaged over windows):
- Top-1 accuracy, top-3 accuracy, macro-F1, weighted-F1.
- Per-genre precision, recall and F1 (`sklearn.metrics.classification_report`).
- A normalized confusion matrix saved as a PNG.
- A comparison table of all models in `reports/results.md`.
- Short written error analysis: which genres get confused and why (e.g. Experimental, Pop and Electronic overlapping).

Optional extra: a UMAP plot of the penultimate-layer embeddings, colored by genre.

---

## 8. Demo (`app/demo.py`)
- Gradio interface: audio upload and microphone input.
- Uses the best model. Resample and preprocess exactly as in training, split into windows, average the probabilities.
- Shows the top 3 genres with probabilities as a bar chart, plus the mel-spectrogram of the input.
- Handles clips shorter than 10 seconds by padding them.
- Run with `python app/demo.py`.

---

## 9. Milestones (2 days)

### Day 1
| # | Milestone | Done when |
|---|---|---|
| 1 | Environment and data | `requirements.txt` installs; fma_metadata + fma_small unzipped; **start the fma_medium download right away** |
| 2 | Metadata and cache on fma_small | `splits.csv` built; mel cache made for fma_small; a dataset sample plotted to check it |
| 3 | CNN end-to-end on fma_small | Model A trains for a few epochs, loss falls, evaluation script runs |
| 4 | Full mel cache on fma_medium | All clips cached, bad files logged, normalization stats saved |
| 5 | CNN on fma_medium | Full training run of Model A started (can run overnight) |

### Day 2
| # | Milestone | Done when |
|---|---|---|
| 6 | MERT model | Embeddings extracted, Model C trained and evaluated |
| 7 | AST model | Model B fine-tuned and evaluated (**skip this if time is short**; A + C is still a complete comparison) |
| 8 | Evaluation report | `reports/results.md` with the comparison table, confusion matrices and error analysis |
| 9 | Demo | Gradio app runs locally with the best model |
| 10 | README | Overview, results, reproduction commands, lessons learned |

**If time runs short, cut in this order:** AST (Model B) → UMAP plot → microphone input in the demo.

---

## 10. Dependencies
```
torch, torchaudio            # CUDA build matching the local driver
transformers, accelerate
librosa, soundfile
numpy, pandas, scikit-learn
pyyaml, tqdm, tensorboard
matplotlib, seaborn
gradio
umap-learn                   # optional
```
System requirement: `ffmpeg` (for MP3 decoding).

---

## 11. Rules and pitfalls
- **Never** evaluate on a random split. Always use FMA's official, artist-separated split.
- Compute normalization statistics on the training split only.
- Each model needs audio at its own sample rate: CNN 22.05 kHz, AST 16 kHz, MERT 24 kHz.
- Cache everything that is expensive (spectrograms, embeddings); never decode MP3s inside the training loop.
- All preprocessing scripts must be resumable and must not crash on a bad file.
- Report macro-F1 alongside accuracy.
- Do not commit data, cached features or checkpoints to git.

---

## 12. References
- Defferrard et al., "FMA: A Dataset For Music Analysis", ISMIR 2017. https://arxiv.org/abs/1612.01840 · https://github.com/mdeff/fma
- Gong et al., "AST: Audio Spectrogram Transformer", Interspeech 2021. https://arxiv.org/abs/2104.01778
- Li et al., "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training", 2023. https://arxiv.org/abs/2306.00107
- Park et al., "SpecAugment", Interspeech 2019. https://arxiv.org/abs/1904.08779
- Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018. https://arxiv.org/abs/1710.09412
- Sturm, "The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use", 2013. https://arxiv.org/abs/1306.1461

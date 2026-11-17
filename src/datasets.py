"""PyTorch datasets + augmentation.

Every dataset returns (x, label):
  train mode: x is one example (e.g. a 10 s crop)
  eval mode:  x has a leading window axis [W, ...]; the engine runs the model on
              every window and averages the probabilities per track.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset

import audio
from utils import EMB_DIR, MEL_DIR, WAV_DIR, mel_stats_path


# ------------------------------------------------------------------ mels (Model A)

class MelDataset(Dataset):
    def __init__(self, df, subset: str, train: bool, cfg: dict):
        d = cfg["data"]
        self.ids = df.track_id.to_numpy()
        self.labels = df.label.to_numpy()
        self.train = train
        self.crop = d.get("crop_frames", audio.WINDOW_FRAMES)
        self.n_windows = d.get("eval_windows", 5)
        stats = json.loads(mel_stats_path(subset).read_text())
        self.mean, self.std = stats["mean"], stats["std"]
        aug = d.get("augment", {}) if train else {}
        self.gain_db = aug.get("gain_db", 0.0)
        self.freq_masks, self.freq_w = aug.get("freq_masks", 0), aug.get("freq_mask_width", 0)
        self.time_masks, self.time_w = aug.get("time_masks", 0), aug.get("time_mask_width", 0)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        # mmap so a training crop reads only ~1/3 of the file
        mel = np.load(MEL_DIR / f"{self.ids[i]}.npy", mmap_mode="r")
        T = mel.shape[1]
        if self.train:
            s = np.random.randint(0, T - self.crop + 1)
            x = np.asarray(mel[:, s:s + self.crop], dtype=np.float32)
            if self.gain_db:
                x = x + np.random.uniform(-self.gain_db, self.gain_db)  # gain = dB offset
            x = (x - self.mean) / self.std
            x = self._spec_augment(x)
            return torch.from_numpy(x)[None], int(self.labels[i])
        starts = audio.window_starts(T, self.crop, self.n_windows)
        x = np.stack([np.asarray(mel[:, s:s + self.crop], dtype=np.float32) for s in starts])
        x = (x - self.mean) / self.std
        return torch.from_numpy(x)[:, None], int(self.labels[i])  # [W, 1, F, T]

    def _spec_augment(self, x):
        F, T = x.shape
        for _ in range(self.freq_masks):
            w = np.random.randint(0, self.freq_w + 1)
            f0 = np.random.randint(0, F - w + 1)
            x[f0:f0 + w] = 0.0  # 0 == mean after normalization
        for _ in range(self.time_masks):
            w = np.random.randint(0, self.time_w + 1)
            t0 = np.random.randint(0, T - w + 1)
            x[:, t0:t0 + w] = 0.0
        return x


# ------------------------------------------------------------- embeddings (Model C)

class EmbeddingDataset(Dataset):
    """Precomputed per-track MERT embeddings (n_layers, 768), held in RAM."""

    def __init__(self, df, subset: str, train: bool, cfg: dict):
        self.labels = df.label.to_numpy()
        self.x = torch.from_numpy(np.stack(
            [np.load(EMB_DIR / f"{t}.npy") for t in df.track_id]).astype(np.float32))
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.x[i]
        return (x if self.train else x[None]), int(self.labels[i])  # eval: W=1


# ------------------------------------------- raw waveform (for AST on days 3-4)

class WaveDataset(Dataset):
    """Waveform crops at any sample rate, resampled on load from the 24 kHz cache.

    cfg["data"]: sample_rate, crop_seconds, eval_windows.
    """

    def __init__(self, df, subset: str, train: bool, cfg: dict):
        d = cfg["data"]
        self.ids = df.track_id.to_numpy()
        self.labels = df.label.to_numpy()
        self.train = train
        self.sr = d["sample_rate"]
        self.crop = int(d.get("crop_seconds", 10.0) * self.sr)
        self.n_windows = d.get("eval_windows", 5)

    def __len__(self):
        return len(self.ids)

    def load(self, i) -> np.ndarray:
        return audio.resample_on_load(np.load(WAV_DIR / f"{self.ids[i]}.npy"), self.sr)

    def __getitem__(self, i):
        y = self.load(i)
        if self.train:
            s = np.random.randint(0, len(y) - self.crop + 1)
            return torch.from_numpy(y[s:s + self.crop].copy()), int(self.labels[i])
        starts = audio.window_starts(len(y), self.crop, self.n_windows)
        return torch.from_numpy(np.stack([y[s:s + self.crop] for s in starts])), int(self.labels[i])


DATASETS = {"mel": MelDataset, "embedding": EmbeddingDataset, "wave": WaveDataset}


def build_dataset(cfg: dict, df, train: bool):
    kind = cfg["data"]["kind"]
    if kind not in DATASETS:
        raise ValueError(f"unknown data.kind {kind!r}; options: {list(DATASETS)}")
    return DATASETS[kind](df, cfg["data"]["subset"], train, cfg)


# ------------------------------------------------------------------- mixup

def mixup(x: torch.Tensor, y_soft: torch.Tensor, alpha: float):
    """Batch-level mixup. y_soft: one-hot/soft targets [B, C]."""
    if alpha <= 0:
        return x, y_soft
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y_soft + (1 - lam) * y_soft[perm]

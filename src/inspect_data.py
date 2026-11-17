"""Sanity-check the caches: plot one track's mel, an augmented training crop,
and its 24 kHz waveform resampled on load to 16 kHz.

    python src/inspect_data.py --subset small [--track-id 2]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import audio  # noqa: E402
from datasets import MelDataset, WaveDataset  # noqa: E402
from utils import MEL_DIR, OUTPUTS, ROOT, load_config, load_splits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="small")
    ap.add_argument("--track-id", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "cnn.yaml", [f"data.subset={args.subset}"])
    df = load_splits(args.subset)
    df = df[df.split == "training"]
    df = df[[(MEL_DIR / f"{t}.npy").exists() for t in df.track_id]]
    row = df[df.track_id == args.track_id] if args.track_id else df.sample(1, random_state=0)
    np.random.seed(0)

    mel = np.load(MEL_DIR / f"{row.track_id.iloc[0]}.npy").astype(np.float32)
    crop, label = MelDataset(row, args.subset, True, cfg)[0]
    wcfg = {"data": {"sample_rate": 16000, "crop_seconds": 10.0, "eval_windows": 3}}
    wav16 = WaveDataset(row, args.subset, False, wcfg).load(0)
    eval_x, _ = MelDataset(row, args.subset, False, cfg)[0]

    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    im = ax[0].imshow(mel, origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax[0], label="dB")
    ax[0].set_title(f"track {row.track_id.iloc[0]} ({row.genre.iloc[0]}): cached log-mel {mel.shape}")
    im = ax[1].imshow(crop[0].numpy(), origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax[1], label="z-score")
    ax[1].set_title(f"augmented training crop {tuple(crop.shape)} (gain, SpecAugment, normalized); label {label}")
    t = np.arange(len(wav16)) / 16000
    ax[2].plot(t, wav16, lw=0.3)
    ax[2].set_title(f"24 kHz cache resampled on load to 16 kHz: {len(wav16)} samples")
    ax[2].set_xlabel("seconds")
    fig.tight_layout()
    out = OUTPUTS / "figures" / f"sample_{args.subset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"mel range [{mel.min():.1f}, {mel.max():.1f}] dB | crop mean {crop.mean():.2f} std {crop.std():.2f} | "
          f"eval windows {tuple(eval_x.shape)} | wav16 {wav16.shape} -> {out}")


if __name__ == "__main__":
    main()

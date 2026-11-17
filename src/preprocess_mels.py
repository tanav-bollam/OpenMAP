"""Single decode pass: MP3 -> (a) 22.05 kHz log-mel cache and (b) 24 kHz waveform cache.

Each MP3 is decoded once at its native rate, then resampled twice:
  data/mels/{track_id}.npy    float16 (128, 1292)   log-mel dB, for the CNN
  data/wav24k/{track_id}.npy  float16 (720000,)     30 s mono at 24 kHz, for MERT
                                                    (and, resampled on load, AST)
Resumable (skips tracks whose outputs exist), writes atomically, never crashes
on a bad file: failures go to data/bad_tracks.txt as "track_id<TAB>reason".

    python src/preprocess_mels.py --subset small
    python src/preprocess_mels.py --subset medium
    python src/preprocess_mels.py --subset medium --stats-only
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import audio
from utils import BAD_TRACKS, MEL_DIR, ROOT, WAV_DIR, load_splits, mel_stats_path, read_bad_tracks, save_json

# Clips with less real audio than this are logged as bad instead of zero-padded.
MIN_SECONDS = 10.0


def _atomic_save(path, arr) -> None:
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def process(item: tuple[int, str]) -> tuple[int, str | None]:
    """Returns (track_id, None) on success or (track_id, reason) on failure."""
    tid, rel_path = item
    mel_out, wav_out = MEL_DIR / f"{tid}.npy", WAV_DIR / f"{tid}.npy"
    if mel_out.exists() and wav_out.exists():
        return tid, None
    try:
        y, sr = audio.decode(str(ROOT / rel_path))
        dur = len(y) / sr if sr else 0.0
        if dur < MIN_SECONDS:
            return tid, f"too short ({dur:.2f}s)"
        if not np.isfinite(y).all():
            return tid, "non-finite samples"

        y24 = audio.pad_or_trim(audio.resample(y, sr, audio.CACHE_SR), audio.CACHE_SAMPLES)
        y22 = audio.pad_or_trim(audio.resample(y, sr, audio.MEL_SR), audio.MEL_SAMPLES)
        mel = audio.logmel(y22)
        assert mel.shape == (audio.N_MELS, audio.MEL_FRAMES), mel.shape

        _atomic_save(wav_out, y24.astype(np.float16))
        _atomic_save(mel_out, mel.astype(np.float16))
        return tid, None
    except Exception as e:  # noqa: BLE001 - must never crash on a bad file
        return tid, f"{type(e).__name__}: {str(e)[:200]}".replace("\n", " ")


def _sums(tid: int) -> tuple[float, float, int]:
    m = np.load(MEL_DIR / f"{tid}.npy").astype(np.float64)
    return m.sum(), np.square(m).sum(), m.size


def compute_stats(subset: str, workers: int) -> None:
    """Global mean/std of log-mel dB values over the TRAINING split only."""
    df = load_splits(subset)
    ids = [t for t in df[df.split == "training"].track_id if (MEL_DIR / f"{t}.npy").exists()]
    s = ss = n = 0
    with Pool(workers) as pool:
        for a, b, c in tqdm(pool.imap_unordered(_sums, ids, chunksize=32), total=len(ids), desc="stats"):
            s, ss, n = s + a, ss + b, n + c
    mean = s / n
    std = float(np.sqrt(ss / n - mean**2))
    save_json({"mean": float(mean), "std": std, "n_tracks": len(ids), "split": "training"},
              mel_stats_path(subset))
    print(f"mel stats ({subset}, {len(ids)} training tracks): mean={mean:.3f} std={std:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="medium")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=0, help="process only the first N tracks (testing)")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    MEL_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    if not args.stats_only:
        df = load_splits(args.subset, drop_bad=False)
        known_bad = read_bad_tracks()
        todo = [(int(t), p) for t, p in zip(df.track_id, df.path)
                if t not in known_bad
                and not ((MEL_DIR / f"{t}.npy").exists() and (WAV_DIR / f"{t}.npy").exists())]
        if args.limit:
            todo = todo[: args.limit]
        print(f"{len(df)} tracks in {args.subset}; {len(todo)} to process "
              f"({len(known_bad)} known bad skipped); {args.workers} workers")

        t0, n_bad = time.time(), 0
        with Pool(args.workers) as pool, open(BAD_TRACKS, "a", encoding="utf-8") as bad_f:
            for tid, reason in tqdm(pool.imap_unordered(process, todo, chunksize=4), total=len(todo)):
                if reason:
                    n_bad += 1
                    bad_f.write(f"{tid}\t{reason}\n")
                    bad_f.flush()
                    tqdm.write(f"bad track {tid}: {reason}")
        print(f"done in {(time.time() - t0) / 60:.1f} min; {n_bad} new bad tracks -> {BAD_TRACKS}")

    if not args.limit:
        compute_stats(args.subset, args.workers)


if __name__ == "__main__":
    main()

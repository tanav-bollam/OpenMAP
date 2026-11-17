"""Audio decoding, resampling and feature extraction shared by preprocessing,
datasets and the demo, so every stage uses exactly the same transforms.

Deliberately torch-free: it is imported by multiprocessing workers.
"""
from __future__ import annotations

import numpy as np

CLIP_SECONDS = 30.0

# Waveform cache: one decode pass writes 24 kHz audio; any other rate is derived
# from it on load (resample_on_load).
CACHE_SR = 24_000
CACHE_SAMPLES = int(CACHE_SR * CLIP_SECONDS)  # 720,000

# Log-mel settings for the CNN (Model A).
MEL_SR = 22_050
MEL_SAMPLES = int(MEL_SR * CLIP_SECONDS)  # 661,500
N_FFT = 2048
HOP = 512
N_MELS = 128
FMIN = 20
FMAX = 11_025
MEL_FRAMES = 1 + MEL_SAMPLES // HOP  # 1292
WINDOW_FRAMES = 431  # ~10 s at 22.05 kHz / hop 512


def decode(path: str) -> tuple[np.ndarray, int]:
    """Decode an audio file to mono float32 at its native sample rate."""
    import librosa

    y, sr = librosa.load(path, sr=None, mono=True)
    return y.astype(np.float32, copy=False), int(sr)


def decode_any(path: str) -> tuple[np.ndarray, int]:
    """Decode user-supplied audio (demo): try libsndfile first, then fall back to the ffmpeg
    binary bundled with the `imageio-ffmpeg` package (handles browser WebM/Opus recordings, M4A...).
    Not used by the training pipeline, so the caches stay bit-identical."""
    try:
        y, sr = decode(path)
        if len(y):
            return y, sr
    except Exception:  # noqa: BLE001 - fall through to ffmpeg
        pass
    import os
    import subprocess
    import tempfile

    import imageio_ffmpeg
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "converted.wav")
        r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", path,
                            "-vn", "-ac", "1", "-ar", "44100", out], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"could not decode this audio file: {r.stderr.strip()[:300]}")
        y, sr = sf.read(out, dtype="float32")
    if len(y) == 0:
        raise RuntimeError("the recording is empty (no audio samples)")
    return y.astype(np.float32, copy=False), int(sr)


def resample(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return y.astype(np.float32, copy=False)
    import soxr

    return soxr.resample(y, sr_from, sr_to, quality="HQ").astype(np.float32, copy=False)


def pad_or_trim(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def logmel(y22k: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram (dB) of 22.05 kHz audio, shape (128, frames), float32."""
    import librosa

    S = librosa.feature.melspectrogram(
        y=y22k, sr=MEL_SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, power=2.0,
    )
    return librosa.power_to_db(S, ref=1.0, top_db=80.0).astype(np.float32)


def resample_on_load(wave_cache: np.ndarray, target_sr: int) -> np.ndarray:
    """Turn a cached 24 kHz waveform into `target_sr` (e.g. 16 kHz for AST)."""
    return resample(np.asarray(wave_cache, dtype=np.float32), CACHE_SR, target_sr)


def window_starts(total: int, window: int, n_windows: int) -> np.ndarray:
    """Evenly spaced window start indices covering [0, total)."""
    if total <= window or n_windows <= 1:
        return np.zeros(1 if n_windows <= 1 else n_windows, dtype=np.int64)
    return np.linspace(0, total - window, n_windows).round().astype(np.int64)

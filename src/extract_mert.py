"""Extract frozen MERT-v1-95M embeddings once per track (fp16, 5 s chunks).

Reads the 24 kHz waveform cache (no MP3 decoding), splits each 30 s clip into
6 x 5 s chunks, runs MERT with output_hidden_states, averages each layer over
time and over the (non-silent) chunks, and saves float32 (13, 768) to
data/embeddings/{track_id}.npy. Resumable.

    python src/extract_mert.py --config configs/mert.yaml --subset small
    python src/extract_mert.py --config configs/mert.yaml
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import EMB_DIR, WAV_DIR, default_num_workers, load_config, load_splits

SILENCE_RMS = 1e-4


class ChunkDataset(Dataset):
    def __init__(self, ids, chunk_samples):
        self.ids, self.chunk = list(ids), chunk_samples

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        y = np.load(WAV_DIR / f"{self.ids[i]}.npy").astype(np.float32)
        n = len(y) // self.chunk
        return torch.from_numpy(y[: n * self.chunk].reshape(n, self.chunk)), self.ids[i]


def collate(batch):
    return torch.cat([b[0] for b in batch]), [b[0].shape[0] for b in batch], [b[1] for b in batch]


def load_mert(mcfg, device):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(mcfg["checkpoint"], trust_remote_code=True,
                                      revision=mcfg.get("revision"))
    return model.to(device).eval()


@torch.no_grad()
def embed_chunks(model, chunks: torch.Tensor, dtype=torch.float16) -> torch.Tensor:
    """chunks [N, samples] at 24 kHz -> per-chunk layer embeddings [N, n_layers, 768] (float32).

    Per-chunk zero-mean/unit-variance normalization, as MERT's feature extractor does.
    """
    x = (chunks - chunks.mean(1, keepdim=True)) / torch.sqrt(chunks.var(1, keepdim=True) + 1e-7)
    with torch.autocast("cuda", dtype=dtype):
        hs = model(x, output_hidden_states=True).hidden_states
    return torch.stack([h.float().mean(1) for h in hs], dim=1)


def pool_track(chunk_emb: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
    """Average chunk embeddings, ignoring silent (e.g. zero-padded) chunks."""
    rms = chunks.pow(2).mean(1).sqrt()
    keep = rms > SILENCE_RMS
    if not keep.any():
        keep[:] = True
    return chunk_emb[keep].mean(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mert.yaml")
    ap.add_argument("--subset", default=None, help="override data.subset")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["mert"]
    subset = args.subset or cfg["data"]["subset"]
    device = torch.device("cuda")
    dtype = torch.float16 if mcfg.get("dtype", "fp16") == "fp16" else torch.bfloat16

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    df = load_splits(subset)
    ids = [t for t in df.track_id if not (EMB_DIR / f"{t}.npy").exists() and (WAV_DIR / f"{t}.npy").exists()]
    missing = sum(not (WAV_DIR / f"{t}.npy").exists() for t in df.track_id)
    if args.limit:
        ids = ids[: args.limit]
    chunk = int(mcfg["chunk_seconds"] * mcfg["sample_rate"])
    tracks_per_batch = max(1, mcfg["chunks_per_batch"] // int(30 // mcfg["chunk_seconds"]))
    print(f"{len(df)} tracks in {subset}; {len(ids)} to embed; {missing} without waveform cache; "
          f"{tracks_per_batch} tracks/batch")

    model = load_mert(mcfg, device)
    dl = DataLoader(ChunkDataset(ids, chunk), batch_size=tracks_per_batch, collate_fn=collate,
                    num_workers=min(4, default_num_workers()), pin_memory=True)
    t0 = time.time()
    for chunks, counts, tids in tqdm(dl, total=len(dl)):
        chunks = chunks.to(device, non_blocking=True)
        emb = embed_chunks(model, chunks, dtype)
        for (e, c), tid in zip(zip(emb.split(counts), chunks.split(counts)), tids):
            out = pool_track(e, c).cpu().numpy().astype(np.float32)
            assert np.isfinite(out).all(), f"non-finite embedding for {tid}"
            tmp = EMB_DIR / f"{tid}.tmp.npy"
            np.save(tmp, out)
            os.replace(tmp, EMB_DIR / f"{tid}.npy")
    print(f"done in {(time.time() - t0) / 60:.1f} min; peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

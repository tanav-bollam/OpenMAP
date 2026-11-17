"""Shared helpers: paths, config loading, seeding, split/genre I/O."""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
MEL_DIR = DATA / "mels"
WAV_DIR = DATA / "wav24k"
EMB_DIR = DATA / "embeddings"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
BAD_TRACKS = DATA / "bad_tracks.txt"


def splits_path(subset: str) -> Path:
    return DATA / f"splits_{subset}.csv"


def genres_path(subset: str) -> Path:
    return DATA / f"genres_{subset}.json"


def mel_stats_path(subset: str) -> Path:
    return DATA / f"mel_stats_{subset}.json"


def load_splits(subset: str, drop_bad: bool = True):
    import pandas as pd

    df = pd.read_csv(splits_path(subset))
    if drop_bad:
        bad = read_bad_tracks()
        df = df[~df.track_id.isin(bad)].reset_index(drop=True)
    return df


def load_genres(subset: str) -> list[str]:
    return json.loads(genres_path(subset).read_text(encoding="utf-8"))


def read_bad_tracks() -> set[int]:
    if not BAD_TRACKS.exists():
        return set()
    out = set()
    for line in BAD_TRACKS.read_text().splitlines():
        if line.strip():
            out.add(int(line.split("\t")[0]))
    return out


# ---------------------------------------------------------------- config

def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        set_by_path(cfg, key, yaml.safe_load(val))
    return cfg


def set_by_path(cfg: dict, dotted: str, value) -> None:
    node = cfg
    *parents, leaf = dotted.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def deep_copy(cfg: dict) -> dict:
    return copy.deepcopy(cfg)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def default_num_workers(cap: int = 6) -> int:
    # Windows uses spawn: every worker re-imports torch (~0.5 GB RAM each),
    # so cap well below cores-2 on a 16 GB machine.
    return max(0, min((os.cpu_count() or 2) - 2, cap))


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

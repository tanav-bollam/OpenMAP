"""Build data/splits_{subset}.csv and data/genres_{subset}.json from tracks.csv.

Uses FMA's official artist-separated split. Never makes a random split.

    python src/prepare_metadata.py --subset small
    python src/prepare_metadata.py --subset medium
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from utils import RAW, genres_path, splits_path

# The ('set','subset') column holds the *smallest* subset a track belongs to,
# and subsets are nested (small ⊂ medium ⊂ large). Note: comparing it as a
# plain string with <= 'medium' would wrongly include 'large' ('l' < 'm').
SUBSETS = {"small": ["small"], "medium": ["small", "medium"]}


def audio_path(subset: str, track_id: int) -> str:
    """Track 2 -> data/raw/fma_medium/000/000002.mp3 (relative to the project root)."""
    tid = f"{track_id:06d}"
    return f"data/raw/fma_{subset}/{tid[:3]}/{tid}.mp3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=SUBSETS, default="medium")
    args = ap.parse_args()

    tracks = pd.read_csv(RAW / "fma_metadata" / "tracks.csv", index_col=0, header=[0, 1])
    sel = tracks[tracks[("set", "subset")].isin(SUBSETS[args.subset])]
    sel = sel[sel[("track", "genre_top")].notna()]

    genres = sorted(sel[("track", "genre_top")].unique())
    label = {g: i for i, g in enumerate(genres)}

    ids = sel.index.to_numpy()
    df = pd.DataFrame({
        "track_id": ids,
        "path": [audio_path(args.subset, i) for i in ids],
        "genre": sel[("track", "genre_top")].to_numpy(),
        "split": sel[("set", "split")].to_numpy(),
    })
    df["label"] = df.genre.map(label)
    df = df[["track_id", "path", "genre", "label", "split"]]
    assert set(df.split) == {"training", "validation", "test"}, set(df.split)

    splits_path(args.subset).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(splits_path(args.subset), index=False)
    genres_path(args.subset).write_text(json.dumps(genres, indent=2))

    print(f"{args.subset}: {len(df)} tracks, {len(genres)} genres -> {splits_path(args.subset)}")
    table = pd.crosstab(df.genre, df.split)[["training", "validation", "test"]]
    table.loc["TOTAL"] = table.sum()
    print(table.sort_values("training", ascending=False).to_string())


if __name__ == "__main__":
    main()

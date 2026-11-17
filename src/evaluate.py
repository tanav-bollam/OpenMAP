"""Evaluate trained runs on the official test split (track level, windows averaged).

    python src/evaluate.py outputs/cnn outputs/mert          # evaluate runs + write reports/results.md
    python src/evaluate.py outputs/cnn_quick --split validation --no-report

Per run, writes to outputs/{run}/: eval_{split}.json, classification_report_{split}.txt,
confusion_{split}.png, probs_{split}.npz.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import engine  # noqa: E402
from datasets import build_dataset  # noqa: E402
from models import build_model  # noqa: E402
from utils import REPORTS, ROOT, default_num_workers, load_splits  # noqa: E402


def load_run(run_dir: Path, device):
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    cfg, genres = ckpt["config"], ckpt["genres"]
    model = build_model(cfg, len(genres)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, genres, ckpt


def evaluate_run(run_dir: Path, split: str, device, umap: bool = False) -> dict:
    model, cfg, genres, ckpt = load_run(run_dir, device)
    df = load_splits(cfg["data"]["subset"])
    df = df[df.split == split]
    ds = build_dataset(cfg, df, train=False)
    nw = min(cfg["train"].get("num_workers", default_num_workers()), default_num_workers())
    dl = DataLoader(ds, batch_size=cfg["train"].get("eval_batch_size", 16), num_workers=nw, pin_memory=True)
    out = engine.predict(model, dl, device, engine.amp_dtype(cfg), return_embed=umap and hasattr(model, "embed"))
    probs, labels = out[0], out[1]
    pred = probs.argmax(1)

    m = engine.metrics(probs, labels)
    m.update(run=run_dir.name, model=cfg["model"]["name"], split=split, n_tracks=int(len(labels)),
             best_epoch=ckpt["epoch"])
    labels_idx = list(range(len(genres)))
    rep_txt = classification_report(labels, pred, labels=labels_idx, target_names=genres, digits=3, zero_division=0)
    m["per_genre"] = classification_report(labels, pred, labels=labels_idx, target_names=genres,
                                           output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, pred, labels=labels_idx)
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    m["top_confusions"] = top_confusions(cm, cm_norm, genres)

    (run_dir / f"classification_report_{split}.txt").write_text(rep_txt)
    (run_dir / f"eval_{split}.json").write_text(json.dumps(m, indent=2))
    np.savez(run_dir / f"probs_{split}.npz", probs=probs, labels=labels, track_id=df.track_id.to_numpy())
    plot_confusion(cm_norm, genres, run_dir / f"confusion_{split}.png", f"{run_dir.name} ({split})")
    if umap and len(out) == 3:
        plot_umap(out[2], labels, genres, run_dir / f"umap_{split}.png", run_dir.name)
    print(f"{run_dir.name} [{split}] acc {m['acc']:.4f} top3 {m['top3_acc']:.4f} "
          f"macro-F1 {m['macro_f1']:.4f} weighted-F1 {m['weighted_f1']:.4f}")
    print(rep_txt)
    return m


def top_confusions(cm, cm_norm, genres, k=8):
    off = [(cm_norm[i, j], int(cm[i, j]), genres[i], genres[j])
           for i in range(len(genres)) for j in range(len(genres)) if i != j and cm[i, j] > 0]
    return [{"true": t, "pred": p, "rate": round(float(r), 3), "count": c}
            for r, c, t, p in sorted(off, reverse=True)[:k]]


def plot_confusion(cm_norm, genres, path, title):
    n = len(genres)
    fig, ax = plt.subplots(figsize=(0.6 * n + 3, 0.55 * n + 2.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, square=True,
                xticklabels=genres, yticklabels=genres, cbar=False, annot_kws={"size": 7}, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Normalized confusion matrix: {title}")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_umap(emb, labels, genres, path, title):
    import umap

    xy = umap.UMAP(n_neighbors=30, min_dist=0.2, random_state=42).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab20")
    for i, g in enumerate(genres):
        s = labels == i
        ax.scatter(xy[s, 0], xy[s, 1], s=4, color=cmap(i % 20), label=g, alpha=0.7)
    ax.legend(markerscale=4, fontsize=7, loc="best")
    ax.set_title(f"UMAP of penultimate embeddings: {title}")
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(results: list[dict]) -> Path:
    REPORTS.mkdir(exist_ok=True)
    lines = ["# Results", "", f"Official FMA test split, track level (window probabilities averaged).", "",
             "| Run | Model | Tracks | Top-1 acc | Top-3 acc | Macro-F1 | Weighted-F1 |",
             "|---|---|---|---|---|---|---|"]
    for m in sorted(results, key=lambda r: -r["macro_f1"]):
        lines.append(f"| {m['run']} | {m['model']} | {m['n_tracks']} | {m['acc']:.3f} | {m['top3_acc']:.3f} | "
                     f"**{m['macro_f1']:.3f}** | {m['weighted_f1']:.3f} |")
    genres = [g for g in results[0]["per_genre"] if g not in ("accuracy", "macro avg", "weighted avg")]
    lines += ["", "## Per-genre F1", "", "| Genre | Support | " + " | ".join(m["run"] for m in results) + " |",
              "|---|---|" + "---|" * len(results)]
    for g in genres:
        sup = int(results[0]["per_genre"][g]["support"])
        lines.append(f"| {g} | {sup} | " + " | ".join(f"{m['per_genre'][g]['f1-score']:.3f}" for m in results) + " |")
    for m in results:
        rel = Path("..") / "outputs" / m["run"] / f"confusion_{m['split']}.png"
        lines += ["", f"## {m['run']}: most frequent confusions", "",
                  f"![confusion matrix]({rel.as_posix()})", "", "| True | Predicted as | Rate | Count |",
                  "|---|---|---|---|"]
        lines += [f"| {c['true']} | {c['pred']} | {c['rate']:.2f} | {c['count']} |" for c in m["top_confusions"]]
    lines += ["", "## Error analysis", "", "_TODO: written analysis._", ""]
    path = REPORTS / "results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories, e.g. outputs/cnn")
    ap.add_argument("--split", default="test", choices=["training", "validation", "test"])
    ap.add_argument("--umap", action="store_true", help="also plot a UMAP of penultimate embeddings")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [evaluate_run(ROOT / r if not Path(r).is_absolute() else Path(r), args.split, device, args.umap)
               for r in args.runs]
    if not args.no_report:
        print(f"report -> {write_report(results)}")


if __name__ == "__main__":
    main()

"""One training script for every model; the YAML config picks model + data.

    python src/train.py --config configs/cnn.yaml --smoke          # 1 batch fwd/bwd, then exit
    python src/train.py --config configs/cnn.yaml --quick          # 3 epochs on fma_small
    python src/train.py --config configs/cnn.yaml                  # full run
    python src/train.py --config configs/cnn.yaml --set train.epochs=5 run_name=foo

Every run (including full runs) starts with a one-batch forward/backward check
that also applies the out-of-memory batch-size fallback.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import engine
from datasets import build_dataset, mixup
from models import build_model
from utils import OUTPUTS, default_num_workers, load_config, load_genres, load_splits, seed_everything


def subsample_training(df, fraction: float, seed: int):
    """Keep a stratified `fraction` of each class in the TRAINING split (validation/test untouched)."""
    import pandas as pd

    parts = [g.sample(n=max(1, round(len(g) * fraction)), random_state=seed)
             for _, g in df[df.split == "training"].groupby("label")]
    return pd.concat(parts + [df[df.split != "training"]]).reset_index(drop=True)


def make_loaders(cfg, df, batch_size):
    t = cfg["train"]
    nw = t.get("num_workers", default_num_workers())
    train_ds = build_dataset(cfg, df[df.split == "training"], train=True)
    val_ds = build_dataset(cfg, df[df.split == "validation"], train=False)
    # Optional class-balanced sampling (train.sampler: none | sqrt | inv). Use it INSTEAD of loss
    # weights (class_weighting: none) so the imbalance correction is not applied twice.
    sw = engine.sample_weights(train_ds.labels, len(load_genres(cfg["data"]["subset"])), t.get("sampler", "none"))
    sampler = (torch.utils.data.WeightedRandomSampler(torch.from_numpy(sw), num_samples=len(sw), replacement=True)
               if sw is not None else None)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
                          drop_last=True, num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    # Validation workers are NOT persistent and capped at 2: on Windows (spawn) each idle
    # persistent worker holds ~0.6 GB, which starved the OS page cache on a 15 GB machine.
    vw = min(nw, t.get("val_num_workers", 2))
    val_dl = DataLoader(val_ds, batch_size=t.get("eval_batch_size", batch_size), shuffle=False,
                        num_workers=vw, pin_memory=True, persistent_workers=False)
    return train_ds, train_dl, val_dl


def train_step(model, x, y, n_classes, cfg, ctx):
    t = cfg["train"]
    y_soft = F.one_hot(y, n_classes).float()
    x, y_soft = mixup(x, y_soft, t.get("mixup_alpha", 0.0))
    with torch.autocast("cuda", dtype=ctx["dtype"]):
        logits = model(x)
    loss = engine.soft_ce(logits, y_soft, ctx["weight"], t.get("label_smoothing", 0.0))
    ctx["opt"].zero_grad(set_to_none=True)
    ctx["scaler"].scale(loss).backward()
    if t.get("grad_clip"):
        ctx["scaler"].unscale_(ctx["opt"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
    ctx["scaler"].step(ctx["opt"])
    ctx["scaler"].update()
    return loss.item(), logits.detach()


def build_optimizer(model, cfg):
    t = cfg["train"]
    groups = model.param_groups(cfg) if hasattr(model, "param_groups") else model.parameters()
    return torch.optim.AdamW(groups, lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4))


def smoke_batch(model, train_ds, cfg, n_classes, ctx, batch_size):
    """One forward/backward pass on a real batch. Raises on OOM."""
    xs, ys = zip(*[train_ds[i] for i in range(batch_size)])
    x = torch.stack(xs).to(ctx["device"])
    y = torch.tensor(ys).to(ctx["device"])
    loss, _ = train_step(model, x, y, n_classes, cfg, ctx)
    if not np.isfinite(loss):
        raise RuntimeError(f"smoke test: non-finite loss {loss}")
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"smoke test OK: batch {tuple(x.shape)} loss {loss:.4f} peak GPU mem {peak:.2f} GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.epochs=3")
    ap.add_argument("--smoke", action="store_true", help="one batch forward/backward, then exit")
    ap.add_argument("--quick", action="store_true", help="3 epochs on fma_small (pre-flight check)")
    args = ap.parse_args()

    overrides = list(args.set)
    if args.quick:
        overrides = ["data.subset=small", "train.epochs=3", "train.early_stop_patience=100"] + overrides
    cfg = load_config(args.config, overrides)
    if args.quick:
        cfg["run_name"] = cfg["run_name"] + "_quick"

    seed_everything(cfg.get("seed", 42))
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA not available - install the CUDA build of torch"

    subset = cfg["data"]["subset"]
    df = load_splits(subset)
    frac = float(cfg["data"].get("train_fraction", 1.0))
    if frac < 1.0:
        df = subsample_training(df, frac, cfg.get("seed", 42))
        print(f"using a stratified {frac:.0%} of the training split: {(df.split == 'training').sum()} tracks")
    genres = load_genres(subset)
    n_classes = len(genres)
    t = cfg["train"]

    model = build_model(cfg, n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dtype = engine.amp_dtype(cfg)
    train_labels = df[df.split == "training"].label.to_numpy()
    weight = engine.class_weights(train_labels, n_classes, t.get("class_weighting", "inv_sqrt"))
    ctx = {
        "device": device, "dtype": dtype,
        "weight": weight.to(device) if weight is not None else None,
        "opt": build_optimizer(model, cfg),
        "scaler": torch.amp.GradScaler("cuda", enabled=dtype == torch.float16),
    }
    print(f"model {cfg['model']['name']}: {n_params / 1e6:.2f}M trainable params | subset {subset} "
          f"({n_classes} classes) | amp {dtype}")

    # ---- pre-flight: one batch forward/backward, with OOM batch-size fallback
    batch_size = t["batch_size"]
    probe_ds = build_dataset(cfg, df[df.split == "training"].head(256), train=True)
    for bs in [batch_size] + [b for b in t.get("fallback_batch_sizes", []) if b < batch_size]:
        try:
            torch.cuda.reset_peak_memory_stats()
            smoke_batch(model, probe_ds, cfg, n_classes, ctx, bs)
            batch_size = bs
            break
        except torch.OutOfMemoryError:
            print(f"OOM at batch size {bs}; trying a smaller one")
            ctx["opt"].zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    else:
        raise RuntimeError("out of memory at every batch size")
    if args.smoke:
        return
    # Fresh model/optimizer so the smoke step doesn't leak into training.
    seed_everything(cfg.get("seed", 42))
    model = build_model(cfg, n_classes).to(device)
    ctx["opt"] = build_optimizer(model, cfg)
    ctx["scaler"] = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    train_ds, train_dl, val_dl = make_loaders(cfg, df, batch_size)
    epochs = t["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        ctx["opt"], max_lr=[g["lr"] for g in ctx["opt"].param_groups], epochs=epochs,
        steps_per_epoch=len(train_dl), pct_start=t.get("warmup_pct", 0.1))

    run_dir = OUTPUTS / cfg["run_name"]
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    cfg["train"]["batch_size_used"] = batch_size
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tb = SummaryWriter(run_dir / "tb")
    log_f = open(run_dir / "metrics.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_top3", "val_macro_f1",
                  "val_weighted_f1", "lr", "seconds"])

    best, best_epoch, patience = -1.0, -1, t.get("early_stop_patience", 8)
    print(f"training {cfg['run_name']}: {len(train_ds)} train tracks, batch {batch_size}, "
          f"{len(train_dl)} steps/epoch, up to {epochs} epochs")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses, correct, seen = [], 0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss, logits = train_step(model, x, y, n_classes, cfg, ctx)
            sched.step()
            losses.append(loss)
            correct += (logits.argmax(1) == y).sum().item()  # vs. pre-mixup labels (approximate)
            seen += y.numel()

        probs, labels = engine.predict(model, val_dl, device, dtype)
        m = engine.metrics(probs, labels)
        w = ctx["weight"].cpu() if ctx["weight"] is not None else None
        val_loss = F.nll_loss(torch.log(torch.from_numpy(probs).clamp_min(1e-8)),
                              torch.from_numpy(labels), weight=w).item()
        lr = ctx["opt"].param_groups[0]["lr"]
        secs = time.time() - t0
        row = [epoch, np.mean(losses), correct / seen, val_loss, m["acc"], m["top3_acc"],
               m["macro_f1"], m["weighted_f1"], lr, secs]
        log.writerow([f"{v:.5g}" if isinstance(v, float) else v for v in row])
        log_f.flush()
        tb.add_scalars("loss", {"train": row[1], "val": val_loss}, epoch)
        tb.add_scalars("acc", {"train": row[2], "val": m["acc"]}, epoch)
        tb.add_scalar("val/macro_f1", m["macro_f1"], epoch)
        tb.add_scalar("lr", lr, epoch)

        improved = m["macro_f1"] > best
        if improved:
            best, best_epoch = m["macro_f1"], epoch
            torch.save({"model": model.state_dict(), "config": cfg, "genres": genres,
                        "epoch": epoch, "val_metrics": m}, run_dir / "best.pt")
        print(f"epoch {epoch:3d} | train loss {row[1]:.4f} acc {row[2]:.3f} | val loss {val_loss:.4f} "
              f"acc {m['acc']:.3f} macro-F1 {m['macro_f1']:.3f} | {secs:.0f}s{' *' if improved else ''}",
              flush=True)
        if epoch - best_epoch >= patience:
            print(f"early stopping: no val macro-F1 improvement for {patience} epochs")
            break

    summary = {"best_val_macro_f1": best, "best_epoch": best_epoch, "epochs_run": epoch,
               "batch_size": batch_size}
    if hasattr(model, "layer_weights"):
        model.load_state_dict(torch.load(run_dir / "best.pt", weights_only=False)["model"])
        summary["layer_weights"] = model.layer_weights()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log_f.close()
    tb.close()
    print(f"best val macro-F1 {best:.4f} at epoch {best_epoch} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

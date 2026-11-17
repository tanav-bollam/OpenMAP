"""Model-agnostic training/eval primitives shared by train.py and evaluate.py."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def amp_dtype(cfg: dict) -> torch.dtype:
    name = cfg["train"].get("amp", "bf16")
    if name == "bf16" and not torch.cuda.is_bf16_supported():
        name = "fp16"
    return AMP_DTYPES[name]


def class_weights(labels: np.ndarray, n_classes: int, mode: str) -> torch.Tensor | None:
    """Loss weights, normalized to mean 1 over classes.

    mode: none | inv_sqrt (weights ∝ 1/sqrt(freq), the default) | inv (weights ∝ 1/freq)."""
    if mode in (None, "none"):
        return None
    powers = {"inv_sqrt": 0.5, "inv": 1.0}
    if mode not in powers:
        raise ValueError(f"unknown class_weighting {mode!r}; options: none, {', '.join(powers)}")
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1) ** powers[mode]
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def sample_weights(labels: np.ndarray, n_classes: int, mode: str) -> np.ndarray | None:
    """Per-example sampling weights for a WeightedRandomSampler: none | sqrt | inv
    (class sampling probability ∝ freq^0.5 or freq^0, i.e. examples weighted 1/sqrt(freq) or 1/freq)."""
    if mode in (None, "none"):
        return None
    powers = {"sqrt": 0.5, "inv": 1.0}
    if mode not in powers:
        raise ValueError(f"unknown sampler {mode!r}; options: none, {', '.join(powers)}")
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    return (1.0 / np.maximum(counts, 1) ** powers[mode])[labels]


def to_device(x, device):
    return x.to(device, non_blocking=True)


@torch.no_grad()
def predict(model, loader, device, dtype, return_embed: bool = False):
    """Track-level probabilities: run every window, average softmax per track.

    Batches from eval datasets are [B, W, ...]; returns probs [N, C], labels [N]
    (and penultimate embeddings [N, D] averaged over windows if requested).
    """
    model.eval()
    probs, labels, embs = [], [], []
    for x, y in loader:
        x = to_device(x, device)
        B, W = x.shape[:2]
        flat = x.reshape(B * W, *x.shape[2:])
        with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
            logits = model(flat)
            if return_embed:
                embs.append(model.embed(flat).float().reshape(B, W, -1).mean(1).cpu())
        p = torch.softmax(logits.float(), dim=-1).reshape(B, W, -1).mean(1)
        probs.append(p.cpu())
        labels.append(y)
    out = (torch.cat(probs).numpy(), torch.cat(labels).numpy())
    return out + (torch.cat(embs).numpy(),) if return_embed else out


def metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, 1)[:, :3]
    return {
        "acc": float(accuracy_score(labels, pred)),
        "top3_acc": float((top3 == labels[:, None]).any(1).mean()),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, pred, average="weighted", zero_division=0)),
    }


def soft_ce(logits, y_soft, weight, label_smoothing):
    """Cross-entropy with soft (mixup) targets, class weights and label smoothing."""
    return F.cross_entropy(logits.float(), y_soft, weight=weight, label_smoothing=label_smoothing)

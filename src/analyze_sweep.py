"""Analyze the imbalance sweep and quantify uncertainty.

Stage 1 (validation only, the test split is NOT read):
    python src/analyze_sweep.py --stage select
  For every variant x seed it evaluates post-hoc prior correction (divide probabilities by
  class_prior**tau, tau in {0, .25, .5, .75, 1}) on the validation set, prints mean +- std
  validation macro-F1, and saves the best (variant, tau) for MERT and the best tau for the CNN
  to outputs/sweep/selection.json.

Stage 2 (reads the test split exactly once per run):
    python src/analyze_sweep.py --stage final
  Evaluates the selected configuration (and the original recipe) on the test split and writes
  reports/imbalance_and_uncertainty.md with seed mean +- std, 95% bootstrap CIs over test tracks,
  and paired-bootstrap CIs for the differences.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

from sweep import VARIANTS
from utils import OUTPUTS, REPORTS, load_genres, load_splits

TAUS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [1, 2, 3, 4, 5]
SWEEP = OUTPUTS / "sweep"
SUBSET = "medium"
N_BOOT = 2000


# ----------------------------------------------------------------------------- helpers

def class_prior() -> np.ndarray:
    df = load_splits(SUBSET)
    n = len(load_genres(SUBSET))
    counts = np.bincount(df[df.split == "training"].label.to_numpy(), minlength=n).astype(np.float64)
    return counts / counts.sum()


def predict(probs: np.ndarray, prior: np.ndarray, tau: float) -> np.ndarray:
    """argmax of p / prior^tau (normalization does not change the argmax)."""
    return (np.log(probs + 1e-12) - tau * np.log(prior)).argmax(1)


def f1_vector(y: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.bincount(y * n + pred, minlength=n * n).reshape(n, n)
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1)  # = 2tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def macro_f1(y, pred, n) -> float:
    return float(f1_vector(y, pred, n).mean())


def load_probs(run_dir: Path, split: str):
    z = np.load(run_dir / f"probs_{split}.npz")
    return z["probs"], z["labels"], z["track_id"]


def mean_std(a) -> str:
    a = np.asarray(a, dtype=float)
    return f"{a.mean():.3f} ± {a.std(ddof=1):.3f}" if len(a) > 1 else f"{a.mean():.3f}"


# ----------------------------------------------------------------------------- stage 1

def stage_select() -> None:
    prior, n = class_prior(), len(load_genres(SUBSET))
    table: dict[str, dict[str, list[float]]] = {}
    for v in VARIANTS:
        runs = [SWEEP / f"{v}_s{s}" for s in SEEDS if (SWEEP / f"{v}_s{s}" / "probs_validation.npz").exists()]
        if not runs:
            continue
        per_tau = {str(t): [] for t in TAUS}
        for r in runs:
            probs, y, _ = load_probs(r, "validation")
            for t in TAUS:
                per_tau[str(t)].append(macro_f1(y, predict(probs, prior, t), n))
        table[v] = per_tau

    print("VALIDATION macro-F1, mean ± std over seeds (rows: variant, columns: tau of prior correction)")
    print(f"{'variant':14s} {'seeds':>5s} " + " ".join(f"tau={t:<9}" for t in TAUS))
    best = (None, None, -1.0)
    for v, per_tau in table.items():
        k = len(per_tau["0.0"])
        print(f"{v:14s} {k:>5d} " + " ".join(f"{mean_std(per_tau[str(t)]):<13}" for t in TAUS))
        for t in TAUS:
            m = float(np.mean(per_tau[str(t)]))
            if m > best[2]:
                best = (v, t, m)

    cnn_probs, cnn_y, _ = load_probs(OUTPUTS / "cnn", "validation")
    cnn_val = {str(t): macro_f1(cnn_y, predict(cnn_probs, prior, t), n) for t in TAUS}
    cnn_tau = float(max(TAUS, key=lambda t: cnn_val[str(t)]))
    print("\nCNN (single seed) validation macro-F1 by tau:",
          {k: round(v, 3) for k, v in cnn_val.items()}, "-> best tau", cnn_tau)
    print(f"\nSELECTED for MERT: variant={best[0]} tau={best[1]} (mean val macro-F1 {best[2]:.3f}); "
          f"baseline = base, tau=0.0 ({np.mean(table['base']['0.0']):.3f})")
    print(f"NOTE: {len(table) * len(TAUS)} configurations were compared on validation, so the winning "
          "validation score is optimistic; the test split has not been read.")

    SWEEP.mkdir(parents=True, exist_ok=True)
    (SWEEP / "selection.json").write_text(json.dumps({
        "mert": {"variant": best[0], "tau": best[1], "val_macro_f1": best[2]},
        "cnn": {"tau": cnn_tau, "val_macro_f1": cnn_val[str(cnn_tau)]},
        "val_table": table, "cnn_val_by_tau": cnn_val, "seeds": SEEDS, "taus": TAUS}, indent=2))
    print("saved", SWEEP / "selection.json")


# ----------------------------------------------------------------------------- stage 2

def ensure_test_probs(run_dir: Path) -> None:
    if (run_dir / "probs_test.npz").exists():
        return
    import torch

    import evaluate

    with contextlib.redirect_stdout(io.StringIO()):
        evaluate.evaluate_run(run_dir, "test", torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def stage_final() -> None:
    sel = json.loads((SWEEP / "selection.json").read_text())
    prior, genres = class_prior(), load_genres(SUBSET)
    n = len(genres)
    v_sel, tau_sel, tau_cnn = sel["mert"]["variant"], sel["mert"]["tau"], sel["cnn"]["tau"]

    mert_orig = "MERT original recipe (inv_sqrt weights, tau=0)"
    configs = {mert_orig: ("base", 0.0)}  # name -> (variant, tau)
    same = (v_sel, tau_sel) == ("base", 0.0)
    if not same:  # only add a second configuration when validation actually picked something different
        configs[f"MERT selected ({v_sel}, tau={tau_sel})"] = (v_sel, tau_sel)

    ref_y = ref_ids = None
    preds: dict[str, list[np.ndarray]] = {}
    for name, (v, tau) in configs.items():
        preds[name] = []
        for s in SEEDS:
            rd = SWEEP / f"{v}_s{s}"
            ensure_test_probs(rd)
            probs, y, ids = load_probs(rd, "test")
            if ref_y is None:
                ref_y, ref_ids = y, ids
            assert (y == ref_y).all() and (ids == ref_ids).all(), "test track order differs between runs"
            preds[name].append(predict(probs, prior, tau))
    # CNN seeds: the original run (seed 42) plus any extra seeds (outputs/cnn_s1, cnn_s2, ...) evaluated on test.
    cnn_dirs = [OUTPUTS / "cnn"] + [d for d in sorted(OUTPUTS.glob("cnn_s[0-9]*")) if (d / "probs_test.npz").exists()]
    cnn_probs_list = []
    for d in cnn_dirs:
        cp, cy, cids = load_probs(d, "test")
        assert (cy == ref_y).all() and (cids == ref_ids).all(), f"test track order differs for {d.name}"
        cnn_probs_list.append(cp)
    cnn_orig = "CNN original recipe (tau=0)"
    preds[cnn_orig] = [predict(cp, prior, 0.0) for cp in cnn_probs_list]
    cnn_same = tau_cnn == 0.0
    if not cnn_same:
        preds[f"CNN selected (tau={tau_cnn})"] = [predict(cp, prior, tau_cnn) for cp in cnn_probs_list]
    print(f"CNN seeds used: {[d.name for d in cnn_dirs]}")
    # AST runs (single seed each; evaluated on test by their own chains)
    ast_names = []
    for label, d in {"AST reduced (8 layers frozen)": OUTPUTS / "ast", "AST full fine-tune": OUTPUTS / "ast_full"}.items():
        if (d / "probs_test.npz").exists():
            ap, ay, aids = load_probs(d, "test")
            assert (ay == ref_y).all() and (aids == ref_ids).all(), f"test track order differs for {d.name}"
            preds[label] = [predict(ap, prior, 0.0)]
            ast_names.append(label)
    y, N = ref_y, len(ref_y)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, N, size=(N_BOOT, N))
    boot: dict[str, np.ndarray] = {}   # per-bootstrap macro-F1 (mean over seeds)
    for name, plist in preds.items():
        boot[name] = np.mean([[macro_f1(y[i], p[i], n) for i in idx] for p in plist], axis=0)

    def ci(a):
        lo, hi = np.percentile(a, [2.5, 97.5])
        return f"[{lo:.3f}, {hi:.3f}]"

    rare = ["Blues", "Easy Listening", "Soul-RnB"]
    rare_i = [genres.index(g) for g in rare]
    lines = ["# Imbalance handling and uncertainty", "",
             "Generated by `src/analyze_sweep.py`. Variants and the prior-correction strength `tau` were chosen on the "
             "**validation** split only; the test split was then read once for the selected and the original configuration.",
             "", "## Setup", "",
             f"- MERT head variants ({len(VARIANTS)}): " + "; ".join(f"`{k}` = loss weights `{a}`, sampler `{b}`" for k, (a, b) in VARIANTS.items()) + ".",
             f"- {len(SEEDS)} seeds per variant ({SEEDS}); the best epoch of each run is chosen by validation macro-F1.",
             f"- Post-hoc prior correction: predict argmax of `p / prior^tau`, tau in {TAUS}, prior = training class frequency.",
             f"- Selected on validation: MERT = `{v_sel}` with tau = {tau_sel}; CNN: tau = {tau_cnn}.",
             f"- {len(VARIANTS) * len(TAUS)} MERT configurations were compared on validation, so validation scores of the winner are optimistic.",
             "", "## Validation sweep (macro-F1, mean ± std over seeds)", "",
             "| Variant | " + " | ".join(f"tau={t}" for t in TAUS) + " |", "|---|" + "---|" * len(TAUS)]
    for v, per_tau in sel["val_table"].items():
        lines.append(f"| {v} | " + " | ".join(mean_std(per_tau[str(t)]) for t in TAUS) + " |")
    lines += ["", "CNN (single seed) validation macro-F1 by tau: " +
              ", ".join(f"tau={k}: {v:.3f}" for k, v in sel["cnn_val_by_tau"].items()), "",
              "## Test results", "",
              "Seed mean ± std is over training seeds. The 95% interval is a bootstrap over the "
              f"{N} test tracks ({N_BOOT} resamples; for multi-seed configs the mean over seeds is bootstrapped), "
              "so it reflects test-set sampling noise, not seed noise. AST rows are single runs, so their seed-to-seed variance is unmeasured.", "",
              "| Configuration | Seeds | Accuracy | Macro-F1 (seed mean ± std) | 95% CI (test tracks) | " +
              " | ".join(f"F1 {g}" for g in rare) + " |", "|---|---|---|---|---|" + "---|" * len(rare)]
    for name, plist in preds.items():
        accs = [float((p == y).mean()) for p in plist]
        f1s = [macro_f1(y, p, n) for p in plist]
        rare_f1 = np.mean([f1_vector(y, p, n)[rare_i] for p in plist], axis=0)
        lines.append(f"| {name} | {len(plist)} | {mean_std(accs)} | {mean_std(f1s)} | {ci(boot[name])} | " +
                     " | ".join(f"{x:.3f}" for x in rare_f1) + " |")

    mert_sel = next((k for k in preds if k.startswith("MERT selected")), None)
    cnn_sel = next((k for k in preds if k.startswith("CNN selected")), None)
    pairs = [("MERT original − CNN original (the headline comparison)", mert_orig, cnn_orig)]
    if mert_sel and cnn_sel:
        pairs.append(("MERT selected − CNN selected", mert_sel, cnn_sel))
    if mert_sel:
        pairs.append(("MERT selected − MERT original (effect of the imbalance fix)", mert_sel, mert_orig))
    if cnn_sel:
        pairs.append(("CNN selected − CNN original (effect of prior correction alone)", cnn_sel, cnn_orig))
    for a in ast_names:
        pairs.append((f"{a} − MERT original", a, mert_orig))
        pairs.append((f"{a} − CNN original", a, cnn_orig))
    if len(ast_names) == 2:
        pairs.append((f"{ast_names[1]} − {ast_names[0]}", ast_names[1], ast_names[0]))
    lines += ["", "## Paired differences in macro-F1 (same bootstrap samples)", "",
              "| Comparison | Difference | 95% CI | Share of resamples > 0 |", "|---|---|---|---|"]
    for label, a, b in pairs:
        d = boot[a] - boot[b]
        pt = np.mean([macro_f1(y, p, n) for p in preds[a]]) - np.mean([macro_f1(y, p, n) for p in preds[b]])
        lines.append(f"| {label} | {pt:+.3f} | {ci(d)} | {(d > 0).mean():.2f} |")
    if same:
        lines += ["", "On validation, none of the tested imbalance fixes (other loss weights, class-balanced sampling, "
                  "post-hoc prior correction) beat the original MERT recipe, so it is the only MERT configuration shown."]
    if cnn_same:
        lines += ["", "For the CNN, post-hoc prior correction did not help on validation (best tau = 0), so it is unchanged."]

    lines += ["", "## Per-genre test F1 (mean over seeds)", "",
              "| Genre | Support | " + " | ".join(k for k in (mert_orig, mert_sel, cnn_orig, *ast_names) if k) + " |",
              "|---|---|" + "---|" * (2 + bool(mert_sel) + len(ast_names))]
    cols = [np.mean([f1_vector(y, p, n) for p in preds[k]], axis=0) for k in (mert_orig, mert_sel, cnn_orig, *ast_names) if k]
    support = np.bincount(y, minlength=n)
    for i, g in enumerate(genres):
        lines.append(f"| {g} | {support[i]} | " + " | ".join(f"{c[i]:.3f}" for c in cols) + " |")
    lines += ["", "_Interpretation: see the notes added to `reports/results.md`._", ""]

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "imbalance_and_uncertainty.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nwrote", out)


def main() -> None:
    import sys

    for stream in (sys.stdout, sys.stderr):  # Windows consoles/redirects default to cp1252
        stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "final"], required=True)
    args = ap.parse_args()
    stage_select() if args.stage == "select" else stage_final()


if __name__ == "__main__":
    main()

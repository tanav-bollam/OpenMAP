"""Measure AST training/eval speed on this GPU and pick a training fraction that fits a time budget.

    python scripts/time_ast.py            # writes outputs/ast_budget.json, last line: FRACTION=<x>

Budget (hours of total training incl. per-epoch validation) comes from env AST_BUDGET_HOURS (default 3).
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from datasets import build_dataset  # noqa: E402
from models import build_model  # noqa: E402
from utils import load_config, load_genres, load_splits, seed_everything  # noqa: E402

cfg = load_config(ROOT / "configs" / "ast.yaml")
if os.environ.get("AST_FREEZE") is not None:      # e.g. AST_FREEZE=0 to time a full fine-tune
    cfg["model"]["freeze_layers"] = int(os.environ["AST_FREEZE"])
seed_everything(cfg.get("seed", 42))
torch.backends.cudnn.benchmark = True
dev = torch.device("cuda")
df = load_splits(cfg["data"]["subset"])
tr, va = df[df.split == "training"], df[df.split == "validation"]
n_classes = len(load_genres(cfg["data"]["subset"]))
ds = build_dataset(cfg, tr.head(32), train=True)
epochs = cfg["train"]["epochs"]
budget = float(os.environ.get("AST_BUDGET_HOURS", "3")) * 3600

result = None
for bs in [cfg["train"]["batch_size"]] + cfg["train"].get("fallback_batch_sizes", []):
    try:
        model = build_model(cfg, n_classes).to(dev)
        opt = torch.optim.AdamW(model.param_groups(cfg), lr=cfg["train"]["lr"])
        xs, ys = zip(*[ds[i] for i in range(bs)])
        x, y = torch.stack(xs).to(dev), torch.tensor(ys).to(dev)
        torch.cuda.reset_peak_memory_stats()
        times = []
        model.train()
        for step in range(6):
            torch.cuda.synchronize(); t = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(x).float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            torch.cuda.synchronize(); times.append(time.time() - t)
        step_time = sum(times[2:]) / len(times[2:])          # skip warm-up steps
        peak_train = torch.cuda.max_memory_allocated() / 1e9
        model.eval(); xe = x.repeat(2, 1)[:12]
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                model(xe)
        torch.cuda.synchronize(); eval_wps = 3 * len(xe) / (time.time() - t)
        result = dict(batch_size=bs, step_time_s=step_time, train_samples_per_s=bs / step_time,
                      eval_windows_per_s=eval_wps, peak_gpu_gb=peak_train)
        break
    except torch.OutOfMemoryError:
        print(f"OOM at batch size {bs}", flush=True)
        del model
        torch.cuda.empty_cache()
if result is None:
    sys.exit("AST does not fit in GPU memory even at the smallest batch size")

# The few timed steps run on a cool GPU, but a long run throttles (84-88 C, clock 2100 -> ~1400-1700 MHz) and
# the data pipeline competes for CPU. Measured on the reduced run: real epoch 1528 s vs 926 s predicted = 1.65x.
slowdown = float(os.environ.get("AST_SLOWDOWN", "1.65"))
n_win = cfg["data"].get("eval_windows", 3)
val_time = slowdown * len(va) * n_win / result["eval_windows_per_s"]
epoch_full = slowdown * len(tr) / result["train_samples_per_s"]
result["assumed_slowdown"] = slowdown
frac = (budget - epochs * val_time) / (epochs * epoch_full)
frac = round(min(1.0, max(0.1, frac)), 3)
result.update(budget_hours=budget / 3600, epochs=epochs, val_time_per_epoch_s=val_time,
              full_epoch_train_time_s=epoch_full, chosen_train_fraction=frac,
              est_total_hours=(epochs * (frac * epoch_full + val_time)) / 3600)
(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "outputs" / "ast_budget.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
epochs_fit = int(budget // (epoch_full + val_time))          # full-data epochs that fit the budget
print(f"EPOCHS={max(3, min(8, epochs_fit))}")
print(f"FRACTION={frac}")
print(f"BATCH={result['batch_size']}")

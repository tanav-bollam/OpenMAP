"""Imbalance sweep for the MERT head: variants x seeds, each trained with train.py and then
evaluated on the VALIDATION split only (the test split is not touched here).

    python src/sweep.py                                  # all variants, seeds 1-5
    python src/sweep.py --variants base sampler_inv --seeds 1

Runs go to outputs/sweep/<variant>_s<seed>/. Resumable: finished runs are skipped.
Then run src/analyze_sweep.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from utils import OUTPUTS, ROOT

# name -> (train.class_weighting, train.sampler)
VARIANTS = {
    "base": ("inv_sqrt", "none"),         # current setup: loss weights ∝ 1/sqrt(freq)
    "noweight": ("none", "none"),         # no imbalance handling (reference)
    "inv": ("inv", "none"),               # loss weights ∝ 1/freq
    "sampler_sqrt": ("none", "sqrt"),     # class-balanced sampling (examples weighted 1/sqrt(freq))
    "sampler_inv": ("none", "inv"),       # fully balanced sampling
}


def run(cmd: list[str], log) -> int:
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    return subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--config", default="configs/mert.yaml")
    args = ap.parse_args()

    (ROOT / "logs").mkdir(exist_ok=True)
    py = sys.executable
    todo = [(v, s) for s in args.seeds for v in args.variants]
    failures = []
    with open(ROOT / "logs" / "sweep.log", "a", encoding="utf-8") as log:
        for n, (v, seed) in enumerate(todo, 1):
            name = f"sweep/{v}_s{seed}"
            run_dir = OUTPUTS / name
            t0 = time.time()
            if not (run_dir / "summary.json").exists():
                cw, sm = VARIANTS[v]
                rc = run([py, "src/train.py", "--config", args.config, "--set", f"run_name={name}",
                          f"seed={seed}", f"train.class_weighting={cw}", f"train.sampler={sm}"], log)
                if rc != 0:
                    failures.append(name)
                    print(f"[{n}/{len(todo)}] {name}: TRAIN FAILED (see logs/sweep.log)", flush=True)
                    continue
            if not (run_dir / "eval_validation.json").exists():
                rc = run([py, "src/evaluate.py", str(run_dir), "--split", "validation", "--no-report"], log)
                if rc != 0:
                    failures.append(name)
                    print(f"[{n}/{len(todo)}] {name}: EVAL FAILED (see logs/sweep.log)", flush=True)
                    continue
            print(f"[{n}/{len(todo)}] {name}: done in {time.time() - t0:.0f}s", flush=True)
    print(f"finished; {len(failures)} failures {failures}")


if __name__ == "__main__":
    main()

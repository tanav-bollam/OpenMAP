#!/usr/bin/env bash
# PRIORITY RUN (planned for tonight): is AST's lead over MERT real, and how good can AST get?
#   1) two more seeds of the reduced AST recipe (8 layers frozen, 3 epochs): outputs/ast_s1, outputs/ast_s2
#   2) one longer run (10 epochs, 8 layers frozen): outputs/ast_e10
#   3) re-run the final analysis (reports/imbalance_and_uncertainty.md, with seed mean +- std for AST)
# Estimated time on this laptop: ~1h25 per 3-epoch run, ~4h20 for the 10-epoch run, ~7h total.
# Resumable: finished runs (summary.json) are skipped, so it is safe to start it again after an interruption.
#   nohup bash scripts/run_ast_seeds.sh > logs/ast_seeds.log 2>&1 &      (or double-click run_ast_seeds.bat)
# Keep the laptop plugged in with sleep off. Nothing else GPU-heavy should run at the same time.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
step() { echo "=== [$(date '+%F %T')] $*"; }
rm -f logs/ast_seeds_done

train_and_eval() {   # $1 = run name, rest = --set overrides
  run=$1; shift
  if [ ! -f outputs/$run/summary.json ]; then
    step "train $run"
    $PY src/train.py --config configs/ast.yaml --set run_name=$run data.train_fraction=1.0 train.batch_size=8 "$@" \
      > logs/${run}_train.log 2>&1 || { echo "TRAIN $run FAILED (see logs/${run}_train.log)"; return 1; }
  fi
  step "evaluate $run (validation + test)"
  $PY src/evaluate.py outputs/$run --split validation --no-report > logs/eval_${run}_val.log 2>&1
  $PY src/evaluate.py outputs/$run --split test --no-report > logs/eval_${run}_test.log 2>&1
}

train_and_eval ast_s1 seed=1
train_and_eval ast_s2 seed=2
train_and_eval ast_e10 train.epochs=10 train.early_stop_patience=4

step "final analysis"
$PY src/analyze_sweep.py --stage final > logs/analysis_final5.log 2>&1 || echo "analysis failed (see logs/analysis_final5.log)"
step "AST SEEDS CHAIN DONE"
touch logs/ast_seeds_done

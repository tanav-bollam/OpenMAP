#!/usr/bin/env bash
# Extra CNN seeds for a fair seed-variance estimate (seed 42 is the original outputs/cnn run).
# Resumable: a finished run (summary.json) is skipped. ~75-90 min per seed on the 6 GB laptop GPU.
#   nohup bash scripts/run_cnn_seeds.sh > logs/cnn_seeds.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
step() { echo "=== [$(date '+%F %T')] $*"; }

step "pre-flight: CNN quick run (3 epochs on fma_small) with the current train.py"
$PY src/train.py --config configs/cnn.yaml --quick || { echo "QUICK TEST FAILED"; exit 1; }

for seed in 1 2 3; do
  run=cnn_s$seed
  if [ ! -f outputs/$run/summary.json ]; then
    step "train $run (seed $seed)"
    $PY src/train.py --config configs/cnn.yaml --set run_name=$run seed=$seed || { echo "TRAIN $run FAILED"; continue; }
  fi
  step "evaluate $run (validation + test)"
  $PY src/evaluate.py outputs/$run --split validation --no-report > logs/eval_${run}_val.log 2>&1
  $PY src/evaluate.py outputs/$run --split test --no-report > logs/eval_${run}_test.log 2>&1
done
step "CNN SEEDS DONE"
touch logs/cnn_seeds_done

#!/usr/bin/env bash
# Runs after the CNN seeds finish (waits for logs/cnn_seeds_done):
#   1) CNN-vs-MERT analysis with all CNN seeds   2) error-analysis tests
#   3) AST: time the GPU, smoke test, quick test, reduced fine-tune, evaluation
#   nohup bash scripts/run_after_cnn.sh > logs/after_cnn.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
step() { echo "=== [$(date '+%F %T')] $*"; }

step "waiting for the CNN seeds"
until [ -f logs/cnn_seeds_done ]; do sleep 30; done

step "1) final analysis with all CNN seeds"
$PY src/analyze_sweep.py --stage final > logs/analysis_final2.log 2>&1 || echo "analysis failed (see logs/analysis_final2.log)"
step "2) error-analysis tests"
(cd src && ../$PY error_analysis.py > ../logs/error_analysis2.log 2>&1) || echo "error analysis failed"

step "3a) AST: measure step time and choose the training fraction"
out=$($PY scripts/time_ast.py 2>&1); echo "$out" | tail -25
FRAC=$(echo "$out" | grep '^FRACTION=' | cut -d= -f2)
BS=$(echo "$out" | grep '^BATCH=' | cut -d= -f2)
[ -z "$FRAC" ] && { echo "AST timing failed; skipping AST"; touch logs/after_cnn_done; exit 1; }

step "3b) AST smoke test (one batch forward/backward)"
$PY src/train.py --config configs/ast.yaml --smoke || { echo "AST SMOKE FAILED"; touch logs/after_cnn_done; exit 1; }

step "3c) AST quick test (1 epoch on a small slice of fma_small)"
$PY src/train.py --config configs/ast.yaml --quick --set train.epochs=1 data.train_fraction=0.04 train.batch_size=$BS \
  > logs/ast_quick.log 2>&1 || { echo "AST QUICK FAILED (see logs/ast_quick.log)"; tail -5 logs/ast_quick.log; touch logs/after_cnn_done; exit 1; }
grep "^epoch\|best val" logs/ast_quick.log | tail -3

step "3d) AST reduced fine-tune: train_fraction=$FRAC, batch=$BS"
$PY src/train.py --config configs/ast.yaml --set data.train_fraction=$FRAC train.batch_size=$BS > logs/ast_train.log 2>&1 \
  || echo "AST TRAINING FAILED (see logs/ast_train.log)"

step "3e) AST evaluation (validation + test)"
$PY src/evaluate.py outputs/ast --split validation --no-report > logs/eval_ast_val.log 2>&1
$PY src/evaluate.py outputs/ast --split test --no-report > logs/eval_ast_test.log 2>&1
step "AFTER-CNN CHAIN DONE"
touch logs/after_cnn_done

#!/usr/bin/env bash
# Second AST run: FULL fine-tune (all 12 layers trainable) for as many epochs as fit the time budget (3-8).
# Starts after the reduced AST run in scripts/run_after_cnn.sh (waits for logs/after_cnn_done).
#   nohup bash scripts/run_ast_full.sh > logs/ast_full_chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
step() { echo "=== [$(date '+%F %T')] $*"; }
step "waiting for chain 2"
until [ -f logs/after_cnn_done ]; do sleep 30; done

step "time a full fine-tune (freeze_layers=0)"
out=$(AST_FREEZE=0 AST_BUDGET_HOURS=${AST_BUDGET_HOURS:-3} $PY scripts/time_ast.py 2>&1); echo "$out" | tail -22
EPOCHS=$(echo "$out" | grep '^EPOCHS=' | cut -d= -f2); BS=$(echo "$out" | grep '^BATCH=' | cut -d= -f2)
[ -z "$EPOCHS" ] && { echo "timing failed; skipping full fine-tune"; touch logs/ast_full_done; exit 1; }

COMMON="run_name=ast_full model.freeze_layers=0 data.train_fraction=1.0 train.epochs=$EPOCHS train.batch_size=$BS"
step "smoke test (full fine-tune, batch $BS)"
$PY src/train.py --config configs/ast.yaml --smoke --set $COMMON || { echo "SMOKE FAILED"; touch logs/ast_full_done; exit 1; }
step "full fine-tune: $EPOCHS epochs"
$PY src/train.py --config configs/ast.yaml --set $COMMON > logs/ast_full_train.log 2>&1 || echo "TRAINING FAILED (see logs/ast_full_train.log)"
step "evaluate ast_full (validation + test)"
$PY src/evaluate.py outputs/ast_full --split validation --no-report > logs/eval_ast_full_val.log 2>&1
$PY src/evaluate.py outputs/ast_full --split test --no-report > logs/eval_ast_full_test.log 2>&1
step "AST FULL CHAIN DONE"
touch logs/ast_full_done

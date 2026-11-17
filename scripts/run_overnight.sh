#!/usr/bin/env bash
# Chain of long jobs. Resumable: each step skips finished work, so re-running is safe.
#   bash scripts/run_overnight.sh > logs/overnight.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
step() { echo "=== [$(date '+%F %T')] $*"; }

# 1. fma_medium extracted + verified?
if [ ! -d data/raw/fma_medium/155 ] && [ -f data/raw/zips/fma_medium.zip ]; then
  step "extracting fma_medium"
  echo "c67b69ea232021025fca9231fc1c7c1a063ab50b  data/raw/zips/fma_medium.zip" | sha1sum -c - || exit 1
  (cd data/raw && /c/Windows/System32/tar.exe -xf zips/fma_medium.zip)
fi

step "metadata (medium)";        $PY src/prepare_metadata.py --subset medium
step "cache medium (mel + 24k)"; $PY src/preprocess_mels.py --subset medium
step "CNN smoke test";           $PY src/train.py --config configs/cnn.yaml --smoke || exit 1
step "CNN full training";        $PY src/train.py --config configs/cnn.yaml
step "CNN evaluation";           $PY src/evaluate.py outputs/cnn --no-report
step "MERT extraction (medium)"; $PY src/extract_mert.py --config configs/mert.yaml
step "MERT quick check";         $PY src/train.py --config configs/mert.yaml --quick
step "MERT full training";       $PY src/train.py --config configs/mert.yaml
step "final evaluation + report"; $PY src/evaluate.py outputs/cnn outputs/mert --umap
step "ALL DONE"

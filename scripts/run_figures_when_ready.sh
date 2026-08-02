#!/usr/bin/env bash

set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
step() { echo "[$(date '+%F %T')] $*"; }

while pgrep -f "run_baseline.py --config" > /dev/null || pgrep -f "run_bayeswarp.py --config" > /dev/null; do
  step "waiting for generation queue"
  sleep 120
done

step "queue drained, exporting"
python scripts/export_range_examples.py --by method --out figures/range_by_method > logs/range_by_method.log 2>&1 \
  && step "ok range_by_method" || step "FAIL range_by_method"
python scripts/export_range_examples.py --by dataset --out figures/range_by_dataset > logs/range_by_dataset.log 2>&1 \
  && step "ok range_by_dataset" || step "FAIL range_by_dataset"
step "FIGURES DONE"

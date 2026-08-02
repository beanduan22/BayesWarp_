#!/usr/bin/env bash

set -u
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
cd "$(dirname "$0")/.."
LOGDIR=logs
mkdir -p "$LOGDIR"

step() { echo "[$(date '+%F %T')] $*"; }

CFG=imagenet_resnet50_smoothgrad
METHODS="adapt nsgen suntest"

if [ ! -d "imagenet/val" ]; then
  step "ABORT: imagenet/val not found"
  exit 1
fi

while pgrep -f "run_baseline.py --config" > /dev/null || pgrep -f "run_bayeswarp.py --config" > /dev/null; do
  step "waiting for in-flight generation jobs"
  sleep 60
done

require_disk() {
  local need_gb="$1"
  local free_gb
  free_gb=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt "$need_gb" ]; then
    step "ABORT: only ${free_gb}G free, need ${need_gb}G"
    exit 1
  fi
}

run() {
  local tag="$1" artifact="$2"; shift 2
  if [ -e "$artifact" ]; then step "skip $tag (have $artifact)"; return 0; fi
  require_disk 10
  step "start $tag"
  if "$@" > "$LOGDIR/$tag.log" 2>&1; then
    step "ok   $tag"
  else
    step "FAIL $tag (see $LOGDIR/$tag.log)"
  fi
}

for b in $METHODS; do
  run "bl_${b}_$CFG" "results/$CFG/failures_$b.pt" \
    python run_baseline.py --config "configs/$CFG.yaml" --baseline "$b"
done

for m in $METHODS; do
  f="results/$CFG/failures_$m.pt"
  [ -f "$f" ] || { step "skip export $CFG/$m (no $f)"; continue; }
  run "export_${CFG}_${m}" "figures/$CFG/$m/scs_strata.json" \
    python scripts/export_scs_strata.py --failures "$f" --out "figures/$CFG/$m"
done

step "ALL DONE"
du -sh results figures 2>/dev/null

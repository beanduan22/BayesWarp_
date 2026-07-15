#!/usr/bin/env bash
# Train missing models, generate test cases with BayesWarp and the baselines,
# then export SCS-stratified original/generated image pairs.
#
# Detached so it survives a dropped connection:
#   setsid nohup bash scripts/run_all_figures.sh > logs/driver.log 2>&1 < /dev/null &
#
# Idempotent: every step is skipped when its output artifact already exists, so
# the script can be re-launched after an interruption.
#
# ImageNet is intentionally absent: it needs the dataset in ./imagenet.

set -u
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
cd "$(dirname "$0")/.."
LOGDIR=logs
mkdir -p "$LOGDIR"

step() { echo "[$(date '+%F %T')] $*"; }

# Do not collide with generation jobs already in flight.
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

for c in mnist_lenet4 mnist_lenet5 cifar10_vgg16 cifar10_resnet18; do
  run "train_$c" "results/$c/best.pt" python train.py --config "configs/${c}_train.yaml"
done

CFGS="mnist_lenet4_smoothgrad mnist_lenet5_smoothgrad cifar10_vgg16_smoothgrad cifar10_resnet18_smoothgrad"
for c in $CFGS; do
  run "bw_$c" "results/$c/failures_main.pt" python run_bayeswarp.py --config "configs/$c.yaml"
  for b in adapt nsgen suntest; do
    run "bl_${b}_$c" "results/$c/failures_$b.pt" \
      python run_baseline.py --config "configs/$c.yaml" --baseline "$b"
  done
done

for c in $CFGS; do
  for m in main adapt nsgen suntest; do
    f="results/$c/failures_$m.pt"
    [ -f "$f" ] || { step "skip export $c/$m (no $f)"; continue; }
    run "export_${c}_${m}" "figures/$c/$m/scs_strata.json" \
      python scripts/export_scs_strata.py --failures "$f" --out "figures/$c/$m"
  done
done

step "ALL DONE"
du -sh results figures 2>/dev/null

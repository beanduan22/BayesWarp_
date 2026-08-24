#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

RUNS="0 1 2 3 4"

MNIST_CONFIGS="mnist_lenet4 mnist_lenet5"
CIFAR_CONFIGS="cifar10_vgg16 cifar10_resnet18"
IMAGENET_CONFIGS="imagenet_vgg19 imagenet_resnet50"
SALIENCY="gradcam ig smoothgrad"

CIFAR_REF=configs/cifar10_resnet18_smoothgrad.yaml
CIFAR_REF_DIR=results/cifar10_resnet18_smoothgrad

NUM_SOURCES=100
PER_SOURCE=10
SCS_THRESHOLD=0.8
FINETUNE_SOURCES="${CIFAR_REF_DIR}/finetune_sources.json"

STORE_SMALL=100
STORE_IMAGENET=40

SCS_PAIRS="mnist_lenet4:adapt mnist_lenet5:suntest cifar10_vgg16:suntest cifar10_resnet18:nsgen imagenet_vgg19:adapt imagenet_resnet50:nsgen"

python train.py --config configs/mnist_lenet4_train.yaml
python train.py --config configs/mnist_lenet5_train.yaml
python train.py --config configs/cifar10_vgg16_train.yaml
python train.py --config configs/cifar10_resnet18_train.yaml

for base in $MNIST_CONFIGS $CIFAR_CONFIGS $IMAGENET_CONFIGS; do
  for saliency in $SALIENCY; do
    python scripts/export_seeds.py \
      --config "configs/${base}_${saliency}.yaml" \
      --out "seeds/${base}_${saliency}.json"
  done
done

python scripts/export_seeds.py \
  --config configs/imagenet_efficientnetb0_smoothgrad.yaml \
  --out seeds/imagenet_efficientnetb0_smoothgrad.json
python scripts/export_seeds.py \
  --config configs/imagenet_resnet50_smoothgrad_300seeds.yaml \
  --out seeds/imagenet_resnet50_smoothgrad_300seeds.json

for base in mnist_lenet4 cifar10_resnet18 imagenet_resnet50; do
  python scripts/export_seeds.py \
    --config "configs/${base}_smoothgrad.yaml" \
    --out "seeds/${base}_smoothgrad_calibration.json" \
    --skip 100 --num-seeds 50
done

for base in $MNIST_CONFIGS $CIFAR_CONFIGS $IMAGENET_CONFIGS; do
  for saliency in $SALIENCY; do
    for run in $RUNS; do
      case "$base" in
        imagenet_*) store=$STORE_IMAGENET ;;
        *)          store=$STORE_SMALL ;;
      esac
      python run_bayeswarp.py --config "configs/${base}_${saliency}.yaml" --run "$run" \
        --store-cases-per-seed "$store"
      python evaluate_results.py \
        --config "configs/${base}_${saliency}.yaml" \
        --failures "results/${base}_${saliency}/failures_main_run${run}.pt" \
        --label "main_run${run}"
    done
  done
done

for base in $MNIST_CONFIGS $CIFAR_CONFIGS $IMAGENET_CONFIGS; do
  python scripts/export_sample_pack.py \
    --config "configs/${base}_smoothgrad.yaml" \
    --failures "results/${base}_smoothgrad/failures_main_run0.pt" \
    --out "samples/${base}_smoothgrad" \
    --run 0 --score-scs --images

  python scripts/export_showcase.py \
    --config "configs/${base}_smoothgrad.yaml" \
    --failures "results/${base}_smoothgrad/failures_main_run0.pt" \
    --out "samples/${base}_smoothgrad/showcase" \
    --count 10 --min-scs 0.8
done

for base in $MNIST_CONFIGS; do
  for baseline in adapt suntest; do
    for run in $RUNS; do
      python run_baseline.py --config "configs/${base}_smoothgrad.yaml" --baseline "$baseline" --run "$run"
      python evaluate_results.py \
        --config "configs/${base}_smoothgrad.yaml" \
        --failures "results/${base}_smoothgrad/failures_${baseline}_run${run}.pt" \
        --label "${baseline}_run${run}"
    done
  done
done

for base in $CIFAR_CONFIGS; do
  for baseline in suntest nsgen; do
    for run in $RUNS; do
      python run_baseline.py --config "configs/${base}_smoothgrad.yaml" --baseline "$baseline" --run "$run"
      python evaluate_results.py \
        --config "configs/${base}_smoothgrad.yaml" \
        --failures "results/${base}_smoothgrad/failures_${baseline}_run${run}.pt" \
        --label "${baseline}_run${run}"
    done
  done
done

for base in $IMAGENET_CONFIGS; do
  for baseline in adapt nsgen; do
    for run in $RUNS; do
      python run_baseline.py --config "configs/${base}_smoothgrad.yaml" --baseline "$baseline" --run "$run"
      python evaluate_results.py \
        --config "configs/${base}_smoothgrad.yaml" \
        --failures "results/${base}_smoothgrad/failures_${baseline}_run${run}.pt" \
        --label "${baseline}_run${run}"
    done
  done
done

for base in $MNIST_CONFIGS $CIFAR_CONFIGS $IMAGENET_CONFIGS; do
  for ablation in no_localization no_bayesian; do
    for run in $RUNS; do
      case "$base" in
        imagenet_*) store=$STORE_IMAGENET ;;
        *)          store=$STORE_SMALL ;;
      esac
      python run_bayeswarp.py --config "configs/${base}_smoothgrad.yaml" --ablation "$ablation" --run "$run" \
        --store-cases-per-seed "$store"
      python evaluate_results.py \
        --config "configs/${base}_smoothgrad.yaml" \
        --failures "results/${base}_smoothgrad/failures_${ablation}_run${run}.pt" \
        --label "${ablation}_run${run}"
    done
  done
done

for ablation in no_merging no_grid; do
  for run in $RUNS; do
    python run_bayeswarp.py --config "$CIFAR_REF" --ablation "$ablation" --run "$run" \
      --store-cases-per-seed "$STORE_SMALL"
    python evaluate_results.py \
      --config "$CIFAR_REF" \
      --failures "${CIFAR_REF_DIR}/failures_${ablation}_run${run}.pt" \
      --label "${ablation}_run${run}"
  done
done

for configuration in svgp_ucb svgp_ei svgp_kappa0 svgp_k1 no_noise exactgp_ucb cmaes; do
  for run in $RUNS; do
    python scripts/run_alternative_search.py \
      --config "$CIFAR_REF" --configuration "$configuration" --run "$run" \
      --num-seeds 50 --budget 2000
  done
done

FINETUNE_PACKS=""
for run in $RUNS; do
  FINETUNE_PACKS="$FINETUNE_PACKS ${CIFAR_REF_DIR}/failures_main_run${run}.pt"
done

python scripts/build_finetune_sources.py \
  --config "$CIFAR_REF" \
  --failures $FINETUNE_PACKS \
  --num-sources "$NUM_SOURCES" \
  --per-source "$PER_SOURCE" \
  --scs-threshold "$SCS_THRESHOLD" \
  --out "$FINETUNE_SOURCES"

for run in $RUNS; do
  for condition in continued standard random autoattack; do
    python scripts/controlled_finetune.py --config "$CIFAR_REF" \
      --condition "$condition" --sources "$FINETUNE_SOURCES" --run "$run"
  done
  python scripts/controlled_finetune.py --config "$CIFAR_REF" --condition high_scs \
    --sources "$FINETUNE_SOURCES" --scs-threshold "$SCS_THRESHOLD" \
    --failures "${CIFAR_REF_DIR}/failures_main_run${run}.pt" --run "$run"
  python scripts/controlled_finetune.py --config "$CIFAR_REF" --condition bayeswarp \
    --sources "$FINETUNE_SOURCES" \
    --failures "${CIFAR_REF_DIR}/failures_main_run${run}.pt" --run "$run"
done

python scripts/verify_finetune_protocol.py \
  --results "${CIFAR_REF_DIR}/controlled_finetune_*.json" \
  --expect-conditions continued standard random autoattack high_scs bayeswarp \
  --out "${CIFAR_REF_DIR}/controlled_finetune_protocol.json"

for base in $MNIST_CONFIGS $CIFAR_CONFIGS $IMAGENET_CONFIGS; do
  python finetune_with_failures.py \
    --config "configs/${base}_smoothgrad.yaml" \
    --failures "results/${base}_smoothgrad/failures_main_run0.pt"
done

for entry in "mnist_lenet5 adapt suntest" "cifar10_resnet18 suntest nsgen" "imagenet_resnet50 adapt nsgen"; do
  set -- $entry
  base="$1"
  shift
  for run in $RUNS; do
    python scripts/compute_dsa.py \
      --config "configs/${base}_smoothgrad.yaml" \
      --failures "results/${base}_smoothgrad/failures_main_run${run}.pt" \
      --label bayeswarp --run "$run"
    for baseline in "$@"; do
      python scripts/compute_dsa.py \
        --config "configs/${base}_smoothgrad.yaml" \
        --failures "results/${base}_smoothgrad/failures_${baseline}_run${run}.pt" \
        --label "$baseline" --run "$run"
    done
  done
done

for entry in "mnist_lenet4 adapt" "cifar10_resnet18 nsgen" "imagenet_resnet50 nsgen"; do
  set -- $entry
  base="$1"
  baseline="$2"
  case "$base" in
    imagenet_*) store=$STORE_IMAGENET ;;
    *)          store=$STORE_SMALL ;;
  esac
  python run_bayeswarp.py --config "configs/${base}_smoothgrad.yaml" \
    --skip 100 --num-seeds 50 --suffix calibration --run 0 \
    --store-cases-per-seed "$store"
  python scripts/compute_redundancy.py \
    --config "configs/${base}_smoothgrad.yaml" \
    --failures "results/${base}_smoothgrad/failures_main_run0.pt" \
    --calibration-failures "results/${base}_smoothgrad/failures_calibration_run0.pt" \
    --label bayeswarp
  python scripts/compute_redundancy.py \
    --config "configs/${base}_smoothgrad.yaml" \
    --failures "results/${base}_smoothgrad/failures_${baseline}_run0.pt" \
    --calibration-failures "results/${base}_smoothgrad/failures_calibration_run0.pt" \
    --label "$baseline"
done

for parameter in alpha rho S m n; do
  python scripts/sensitivity_sweep.py --config "$CIFAR_REF" --parameter "$parameter"
done

python scripts/runtime_breakdown.py --config "$CIFAR_REF"
python scripts/runtime_breakdown.py --config configs/imagenet_resnet50_smoothgrad.yaml

python scripts/plot_tsne.py --config configs/mnist_lenet5_smoothgrad.yaml \
  --failures results/mnist_lenet5_smoothgrad/failures_main_run0.pt --out figures/lenet5.pdf
python scripts/plot_tsne.py --config "$CIFAR_REF" \
  --failures "${CIFAR_REF_DIR}/failures_main_run0.pt" --out figures/resnet18.pdf
python scripts/plot_tsne.py --config configs/imagenet_resnet50_smoothgrad.yaml \
  --failures results/imagenet_resnet50_smoothgrad/failures_main_run0.pt --out figures/resnet50.pdf

for run in $RUNS; do
  python run_bayeswarp.py --config configs/imagenet_efficientnetb0_smoothgrad.yaml --run "$run" \
    --store-cases-per-seed "$STORE_IMAGENET"
  python evaluate_results.py --config configs/imagenet_efficientnetb0_smoothgrad.yaml \
    --failures "results/imagenet_efficientnetb0_smoothgrad/failures_main_run${run}.pt" \
    --label "main_run${run}"
  python run_baseline.py --config configs/imagenet_efficientnetb0_smoothgrad.yaml --baseline nsgen --run "$run"
  python evaluate_results.py --config configs/imagenet_efficientnetb0_smoothgrad.yaml \
    --failures "results/imagenet_efficientnetb0_smoothgrad/failures_nsgen_run${run}.pt" \
    --label "nsgen_run${run}"

  python run_bayeswarp.py --config configs/imagenet_resnet50_smoothgrad_300seeds.yaml --run "$run" \
    --store-cases-per-seed "$STORE_IMAGENET"
  python run_baseline.py --config configs/imagenet_resnet50_smoothgrad_300seeds.yaml --baseline nsgen --run "$run"
done

python scripts/export_sample_pack.py \
  --config configs/imagenet_efficientnetb0_smoothgrad.yaml \
  --failures results/imagenet_efficientnetb0_smoothgrad/failures_main_run0.pt \
  --out samples/imagenet_efficientnetb0_smoothgrad \
  --run 0 --score-scs --images

python scripts/export_showcase.py \
  --config configs/imagenet_efficientnetb0_smoothgrad.yaml \
  --failures results/imagenet_efficientnetb0_smoothgrad/failures_main_run0.pt \
  --out samples/imagenet_efficientnetb0_smoothgrad/showcase \
  --count 10 --min-scs 0.8

sample_human_validation() {
  base="$1"
  shift
  packs=""
  methods=""
  runs=""
  for spec in "$@"; do
    label="${spec%%:*}"
    prefix="${spec##*:}"
    for run in $RUNS; do
      packs="$packs results/${base}_smoothgrad/failures_${prefix}_run${run}.pt"
      methods="$methods $label"
      runs="$runs $run"
    done
  done
  python scripts/sample_human_validation.py \
    --config "configs/${base}_smoothgrad.yaml" \
    --failures $packs \
    --methods $methods \
    --runs $runs \
    --per-method 30 \
    --out-dir "results/human_validation/${base}"
}

for base in $MNIST_CONFIGS; do
  sample_human_validation "$base" bayeswarp:main adapt:adapt suntest:suntest
done

for base in $CIFAR_CONFIGS; do
  sample_human_validation "$base" bayeswarp:main suntest:suntest nsgen:nsgen
done

for base in $IMAGENET_CONFIGS; do
  sample_human_validation "$base" bayeswarp:main adapt:adapt nsgen:nsgen
done

mkdir -p results/statistics
for pair in $SCS_PAIRS; do
  base="${pair%%:*}"
  baseline="${pair##*:}"
  python scripts/compute_statistics.py paired-seed \
    --a "results/${base}_smoothgrad/evaluation_main_run*.json" \
    --b "results/${base}_smoothgrad/evaluation_${baseline}_run*.json" \
    --metric scs \
    --allow-seed-intersection \
    --out "results/statistics/scs_${base}_bayeswarp_vs_${baseline}.json"
done

python scripts/aggregate_results.py --root results --out results/summary.csv

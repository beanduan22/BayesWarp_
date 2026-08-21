# BayesWarp

Bayesian-guided white-box testing of neural network image classifiers. Mutations
are localized to decision-critical input regions via saliency analysis, then
explored with uncertainty-aware Bayesian optimization over a low-dimensional
grid parameterization under a fixed target-model evaluation budget.


## Install

```bash
conda create -n bayeswarp python=3.10 -y
conda activate bayeswarp
pip install -r requirements.txt
pip install torchvision torchmetrics torch-fidelity open-clip-torch
pip install -e .
```

## Datasets

MNIST and CIFAR-10 download automatically to `./data` on first run.

ImageNet must be provided manually in `./imagenet/` with this layout:

```
imagenet/
  train/<class>/*.JPEG
  val/<class>/*.JPEG
```

Download from https://image-net.org/ after registering. Arrange `val/` into class subfolders using the official devkit or the script at https://github.com/soumith/imagenetloader.torch.

## Trained model weights

`weights/` holds the four checkpoints. They are stored with Git LFS, so a plain
`git clone` fetches pointer files; run `git lfs pull` to get the checkpoints
themselves.

| File | Model | Dataset | Size |
|---|---|---|---|
| `weights/mnist_lenet4.pt` | LeNet-4 | MNIST | 0.1 MB |
| `weights/mnist_lenet5.pt` | LeNet-5 | MNIST | 0.2 MB |
| `weights/cifar10_resnet18.pt` | ResNet18 | CIFAR-10 | 42.7 MB |
| `weights/cifar10_vgg16.pt` | VGG16 | CIFAR-10 | 512.3 MB |

Verify them, then place each checkpoint at the path its config expects:

```bash
git lfs pull
sha256sum -c weights.sha256

mkdir -p results/cifar10_resnet18
cp weights/cifar10_resnet18.pt results/cifar10_resnet18/best.pt
```

The ImageNet subjects (VGG19-BN, ResNet50, EfficientNet-B0) use the torchvision
pretrained weights, so their configs set `checkpoint: null` and need no asset.

Every checkpoint is reproducible from this repository without the assets:

```bash
python train.py --config configs/cifar10_resnet18_train.yaml
```

## Run

Train, generate failures, evaluate, then fine-tune. Swap the config name for any entry in `configs/`.

```bash
python train.py --config configs/mnist_lenet5_train.yaml

python run_bayeswarp.py --config configs/mnist_lenet5_smoothgrad.yaml

python evaluate_results.py \
  --config configs/mnist_lenet5_smoothgrad.yaml \
  --failures results/mnist_lenet5_smoothgrad/failures_main_run0.pt

python finetune_with_failures.py \
  --config configs/mnist_lenet5_smoothgrad.yaml \
  --failures results/mnist_lenet5_smoothgrad/failures_main_run0.pt
```

Ablations:

```bash
python run_bayeswarp.py --config configs/mnist_lenet5_smoothgrad.yaml --ablation no_localization
python run_bayeswarp.py --config configs/mnist_lenet5_smoothgrad.yaml --ablation no_bayesian
```

Baselines (ADAPT, NSGen, SUNTest), run under the same seeds, oracle, and budget.
Output is consumed by `evaluate_results.py` and `finetune_with_failures.py`
unchanged. See [baselines/README.md](baselines/README.md).

```bash
python run_baseline.py --config configs/mnist_lenet5_smoothgrad.yaml --baseline adapt
```

Reproduce everything:

```bash
bash scripts/reproduce_paper.sh
```

## Controlled fine-tuning study

All six conditions add training samples derived from the same source samples:
same identities, same count, same class distribution. Only the derivation
differs. That is enforced by a source manifest built once, before any condition
runs, and recorded in every result file.

```bash
# 100 shared sources, 10 added samples derived from each -> 1000 per condition.
# A source is eligible only if every failure pack the generated conditions draw
# on supplies at least 10 cases for it, and at least 10 above the SCS threshold.
python scripts/build_finetune_sources.py \
  --config configs/cifar10_resnet18_smoothgrad.yaml \
  --failures results/cifar10_resnet18_smoothgrad/failures_main_run{0,1,2,3,4}.pt \
  --num-sources 100 --per-source 10 --scs-threshold 0.8 \
  --out results/cifar10_resnet18_smoothgrad/finetune_sources.json

python scripts/controlled_finetune.py \
  --config configs/cifar10_resnet18_smoothgrad.yaml \
  --condition continued \
  --sources results/cifar10_resnet18_smoothgrad/finetune_sources.json \
  --run 0

# Then check that every condition really used the same manifest.
python scripts/verify_finetune_protocol.py \
  --results "results/cifar10_resnet18_smoothgrad/controlled_finetune_*.json" \
  --expect-conditions continued standard random autoattack high_scs bayeswarp
```

Per condition, the `per_source` samples derived from one source are:

| Condition | Derivation | Distinct samples per source |
|---|---|---|
| `continued` | the unmodified source, repeated | 1 by construction: a budget-matched control that adds no new information |
| `standard` | independent draws from the dataset's standard augmentation | up to `per_source` (1 on MNIST, whose standard augmentation is the identity) |
| `random` | independent uniform perturbations inside the same critical region | up to `per_source` |
| `autoattack` | AutoAttack rounds with different attack seeds | often fewer than `per_source`; AutoAttack is only partly randomised |
| `bayeswarp` | generated cases for that source, round-robin over predicted class | up to `per_source` |
| `high_scs` | the same, restricted to cases above the SCS threshold | up to `per_source` |

Each result file records the realised `distinct_per_source`, so a condition that
collapses onto duplicates is visible in the results rather than assumed away.

Source images are read through the deterministic evaluation transform, the same
one the generation seeds use. Reading them from the training dataset would apply
a random crop and flip to every condition, which would confound the comparison.

## Statistics

Per-seed metrics carry the dataset-level seed id that produced them
(`seed_ids`, and `scs_seed_ids` for SCS), and `scripts/compute_statistics.py`
pairs by that id. A seed present on one side and not the other is an error, not
a silent truncation to the shorter array; pass `--allow-seed-intersection` to
compare the shared seeds and have the number dropped recorded in the output.

```bash
python scripts/compute_statistics.py paired-seed \
  --a "results/cifar10_resnet18_smoothgrad/evaluation_main_run*.json" \
  --b "results/cifar10_resnet18_smoothgrad/evaluation_nsgen_run*.json" \
  --metric scs --out results/statistics/scs_cifar10_resnet18.json

# One Holm correction over the family of six SCS settings.
python scripts/compute_statistics.py holm-family \
  --inputs results/statistics/scs_*.json \
  --labels mnist_lenet4 mnist_lenet5 cifar10_vgg16 cifar10_resnet18 imagenet_vgg19 imagenet_resnet50 \
  --family-name "SCS: BayesWarp vs strongest baseline" \
  --out results/statistics/scs_holm_family.json
```

Every paired result reports both estimands with the test that matches each:

* `mean_difference` with `mean_difference_ci_*` and `mean_difference_p_value` (paired t);
* `hodges_lehmann_shift` with `hodges_lehmann_ci_*` and `wilcoxon_p_value` (signed rank).

`primary_test` and `primary_estimate` name which pair the paper reports, so a
t-interval is never quoted next to a Wilcoxon p-value.

Per-seed SCS is scored on a per-seed stratified draw (`--scs-per-seed`, default
20) rather than a global subsample, so it is defined for every seed that
produced a case and comparable across methods. Seeds that produced no cases have
no SCS and are listed in `scs_seeds_without_cases`.

## Human validation

The sample is drawn across all runs, with at most one case per source seed and a
rotation over source classes, so 30 annotated cases per method are 30 distinct
sources rather than a cluster of repeats from a few prolific seeds.

```bash
python scripts/sample_human_validation.py \
  --config configs/cifar10_resnet18_smoothgrad.yaml \
  --failures results/cifar10_resnet18_smoothgrad/failures_main_run{0,1,2,3,4}.pt \
  --methods bayeswarp bayeswarp bayeswarp bayeswarp bayeswarp \
  --runs 0 1 2 3 4 \
  --per-method 30 --max-per-seed 1 \
  --out-dir results/human_validation/cifar10_resnet18

python scripts/human_validation_report.py \
  --annotations results/human_validation/cifar10_resnet18/annotations.json \
  --key results/human_validation/cifar10_resnet18/key.json \
  --out results/human_validation/cifar10_resnet18/report.json
```

`sampling.json` records the realised coverage (distinct seeds, cases per seed,
runs, classes). The report always computes a cluster bootstrap over source seeds
alongside the Wilson interval and sets `any_clustering`; report the bootstrap
interval whenever a seed contributed more than one annotated case, since Wilson
assumes it did not.

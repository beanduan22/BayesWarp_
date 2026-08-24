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
`torch.hub` fetches each on first use into `~/.cache/torch/hub/checkpoints/`
(override the root with `TORCH_HOME`) and verifies it against the sha256 prefix
in the filename.

| Config | torchvision weights enum | Download |
|---|---|---|
| `imagenet_vgg19_*` | `VGG19_BN_Weights.IMAGENET1K_V1` | https://download.pytorch.org/models/vgg19_bn-c79401a0.pth |
| `imagenet_resnet50_*` | `ResNet50_Weights.IMAGENET1K_V1` | https://download.pytorch.org/models/resnet50-0676ba61.pth |
| `imagenet_efficientnetb0_smoothgrad` | `EfficientNet_B0_Weights.IMAGENET1K_V1` | https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth |

The CIFAR-10 subjects are initialised from `VGG16_Weights.IMAGENET1K_V1`
(https://download.pytorch.org/models/vgg16-397923af.pth) and
`ResNet18_Weights.IMAGENET1K_V1`
(https://download.pytorch.org/models/resnet18-f37072fd.pth) before their own
checkpoint is loaded over the top.

On an offline machine, place the files under
`~/.cache/torch/hub/checkpoints/` under their original names:

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
cd ~/.cache/torch/hub/checkpoints
wget https://download.pytorch.org/models/vgg19_bn-c79401a0.pth
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
wget https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth
```

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
unchanged.

```bash
python run_baseline.py --config configs/mnist_lenet5_smoothgrad.yaml --baseline adapt
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



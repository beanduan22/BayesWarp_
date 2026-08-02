# BayesWarp

Bayesian-guided white-box testing of neural network image classifiers. Mutations
are localized to decision-critical input regions via saliency analysis, then
explored with uncertainty-aware Bayesian optimization over a low-dimensional
grid parameterization under a fixed target-model evaluation budget.

## Hyperparameters

The `bayeswarp` block of each config:

| Config key | Symbol | Meaning |
|---|---|---|
| `alpha` | `α` | Proportion of salient pixels retained |
| `area_min` | `A_min` | Minimum connected-component area |
| `tau_iou` | `τ_iou` | Box-merging IoU threshold |
| `d_max` | `d_max` | Box-merging centroid distance |
| `rho` | `ρ` | Max share of the image allowed for mutation |
| `S` | `S` | Candidate increments sampled per search step |
| `eta` | `η` | Pixel-range relaxation factor |
| `r` | `r` | Max grid-level mutation magnitude; `u` is clipped to `[-r, r]` |
| `beta_min`, `beta_max` | `β` | Stagnation noise magnitude, sampled uniformly from this range |
| `epsilon` | `ε` | Stagnation threshold on the objective change |
| `kappa` | `κ` | UCB exploration weight |
| `n` | `n` | Mutation grid is `n × n` per channel; search dim is `channels · n²` |
| `m` | `m` | Max inducing points; the active count is `min(m, |D_tg|)` |
| `budget` | `B` | Total target-model evaluations **per seed**, split across targets |
| `max_target_classes` | `K` | Alternative target classes explored per seed |

`budget` is the total per seed, not per target: it is divided as evenly as
possible across the `K` confidence-ranked target classes, with any remainder
assigned by confidence rank. Every forward pass on a newly constructed input
counts against it, including the extra evaluation after a stagnation
perturbation. The seed's own prediction is cached and not counted; saliency
computation is excluded from the budget but included in wall-clock time.

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

The ImageNet subjects (VGG19, ResNet50, EfficientNet-B0) use the torchvision
pretrained weights, so their configs set `checkpoint: null` and need no asset.

Every checkpoint is reproducible from this repository without the assets:

```bash
python train.py --config configs/cifar10_resnet18_train.yaml
```

## Run

Train, generate failures, evaluate, then fine-tune. Swap the config name for any entry in `configs/` (MNIST / CIFAR-10 / ImageNet × LeNet-4 / LeNet-5 / VGG16 / VGG19 / ResNet18 / ResNet50 × Grad-CAM / Integrated Gradients / SmoothGrad).

```bash
python train.py --config configs/mnist_lenet5_train.yaml

python run_bayeswarp.py --config configs/mnist_lenet5_smoothgrad.yaml

python evaluate_results.py \
  --config configs/mnist_lenet5_smoothgrad.yaml \
  --failures results/mnist_lenet5_smoothgrad/failures_main.pt

python finetune_with_failures.py \
  --config configs/mnist_lenet5_smoothgrad.yaml \
  --failures results/mnist_lenet5_smoothgrad/failures_main.pt
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

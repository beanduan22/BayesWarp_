# BayesWarp

A cleaned, GitHub-ready experimental codebase for the paper:

**Testing Neural Networks via Bayesian-Guided Exploration of Decision Landscapes**

This repository implements the **full BayesWarp pipeline** from the paper draft you provided:
- decision-critical region localization from saliency maps,
- grid-parameterized localized mutation,
- Bayesian-guided search with an inducing-point GP surrogate,
- adaptive target-class scheduling,
- ablations: `w/o Localization`, `w/o Bayesian`,
- model training, failure generation, evaluation, and fine-tuning.

It also includes:
- reproducible YAML configs,
- smoke tests,
- plotting/aggregation helpers,
- baseline adapter stubs for plugging in official ADAPT / NSGen / SUNTest repos.

## What is fully implemented here

- **BayesWarp-C** (Grad-CAM)
- **BayesWarp-I** (Integrated Gradients)
- **BayesWarp-S** (SmoothGrad)
- **Training** for LeNet-4 / LeNet-5 / VGG16 / VGG19 / ResNet18 / ResNet50
- **Failure generation** under the paper's testing pipeline
- **Fine-tuning with generated failures**
- **Metrics**: NoF, FSR, TPF, DoF, NC, TKNC, CNC
- **Optional metrics**: FID and SCS (require optional dependencies)

## What is intentionally left as adapters

Your paper text does **not** contain enough algorithmic detail to claim exact re-implementations of:
- ADAPT
- NSGen
- SUNTest

So this repo includes **adapter interfaces** instead of pretending those baselines are fully reproduced.
That keeps the GitHub repo honest and publication-safe.

## Environment

```bash
conda create -n bayeswarp python=3.10 -y
conda activate bayeswarp
pip install -r requirements.txt
pip install -e .
```

For CIFAR-10 / ImageNet / optional FID / optional SCS, also install:

```bash
pip install torchvision open-clip-torch torchmetrics
```

## Repository structure

```text
bayeswarp_repo_verified/
├── configs/
├── scripts/
├── src/bayeswarp/
│   ├── baselines/
│   ├── bo/
│   ├── data/
│   ├── interpretability/
│   ├── localization/
│   ├── metrics/
│   ├── models/
│   ├── mutation/
│   ├── testing/
│   └── utils/
├── tests/
├── train.py
├── run_bayeswarp.py
├── evaluate_results.py
├── finetune_with_failures.py
└── requirements.txt
```

## Quick start

### 1) Train
```bash
python train.py --config configs/mnist_lenet5_train.yaml
```

### 2) Generate BayesWarp failures
```bash
python run_bayeswarp.py --config configs/mnist_lenet5_smoothgrad.yaml
```

### 3) Run ablations
```bash
python run_bayeswarp.py --config configs/mnist_lenet5_smoothgrad.yaml --ablation no_localization
python run_bayeswarp.py --config configs/mnist_lenet5_smoothgrad.yaml --ablation no_bayesian
```

### 4) Evaluate
```bash
python evaluate_results.py --config configs/mnist_lenet5_smoothgrad.yaml   --failures results/mnist_lenet5_smoothgrad/failures_main.pt
```

### 5) Fine-tune
```bash
python finetune_with_failures.py --config configs/mnist_lenet5_smoothgrad.yaml   --failures results/mnist_lenet5_smoothgrad/failures_main.pt
```

## Reproduce the full paper pipeline

See:
- `scripts/reproduce_paper.sh`
- `scripts/aggregate_results.py`
- `scripts/plot_metrics.py`

## Validation

This cleaned version has been structured to avoid empty folders and placeholder files.
Run the built-in validation:

```bash
python scripts/verify_repo.py
```

## Citation

```bibtex
@misc{bayeswarp2026,
  title={Testing Neural Networks via Bayesian-Guided Exploration of Decision Landscapes},
  author={Bin Duan and Meiru Che and Guowei Yang},
  year={2026}
}
```

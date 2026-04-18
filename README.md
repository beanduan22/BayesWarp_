# BayesWarp

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

Reproduce everything:

```bash
bash scripts/reproduce_paper.sh
```

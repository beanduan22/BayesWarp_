#!/usr/bin/env bash
set -e

python train.py --config configs/mnist_lenet4_train.yaml
python train.py --config configs/mnist_lenet5_train.yaml
python train.py --config configs/cifar10_vgg16_train.yaml
python train.py --config configs/cifar10_resnet18_train.yaml
python train.py --config configs/imagenet_vgg19_train.yaml
python train.py --config configs/imagenet_resnet50_train.yaml

python run_bayeswarp.py --config configs/mnist_lenet4_gradcam.yaml
python run_bayeswarp.py --config configs/mnist_lenet4_ig.yaml
python run_bayeswarp.py --config configs/mnist_lenet4_smoothgrad.yaml
python run_bayeswarp.py --config configs/mnist_lenet5_gradcam.yaml
python run_bayeswarp.py --config configs/mnist_lenet5_ig.yaml
python run_bayeswarp.py --config configs/mnist_lenet5_smoothgrad.yaml
python run_bayeswarp.py --config configs/cifar10_vgg16_gradcam.yaml
python run_bayeswarp.py --config configs/cifar10_vgg16_ig.yaml
python run_bayeswarp.py --config configs/cifar10_vgg16_smoothgrad.yaml
python run_bayeswarp.py --config configs/cifar10_resnet18_gradcam.yaml
python run_bayeswarp.py --config configs/cifar10_resnet18_ig.yaml
python run_bayeswarp.py --config configs/cifar10_resnet18_smoothgrad.yaml
python run_bayeswarp.py --config configs/imagenet_vgg19_gradcam.yaml
python run_bayeswarp.py --config configs/imagenet_vgg19_ig.yaml
python run_bayeswarp.py --config configs/imagenet_vgg19_smoothgrad.yaml
python run_bayeswarp.py --config configs/imagenet_resnet50_gradcam.yaml
python run_bayeswarp.py --config configs/imagenet_resnet50_ig.yaml
python run_bayeswarp.py --config configs/imagenet_resnet50_smoothgrad.yaml

python run_bayeswarp.py --config configs/mnist_lenet5_smoothgrad.yaml --ablation no_localization
python run_bayeswarp.py --config configs/mnist_lenet5_smoothgrad.yaml --ablation no_bayesian

python scripts/aggregate_results.py --root results --out results/summary.csv

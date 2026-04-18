from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Subset


_DATASET_META = {
    'mnist': {'num_classes': 10, 'channels': 1, 'size': 28},
    'cifar10': {'num_classes': 10, 'channels': 3, 'size': 32},
    'imagenet': {'num_classes': 1000, 'channels': 3, 'size': 224},
}


def dataset_meta(name: str) -> Dict:
    return _DATASET_META[name.lower()]


def _tv():
    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise ImportError(
            'torchvision is required for MNIST/CIFAR-10/ImageNet dataset loading. '
            'Install a torchvision build compatible with your torch version.'
        ) from e
    return datasets, transforms


def build_transforms(name: str, normalization: str = 'none', image_size: Optional[int] = None):
    datasets, transforms = _tv()
    name = name.lower()
    mean, std = None, None
    if normalization == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    def maybe_norm(tf_list, channels: int):
        if mean is not None and channels == 3:
            tf_list.append(transforms.Normalize(mean=mean, std=std))
        return tf_list

    if name == 'mnist':
        size = image_size or 28
        train_tf = transforms.Compose(maybe_norm([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ], 1))
        test_tf = transforms.Compose(maybe_norm([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ], 1))
    elif name == 'cifar10':
        size = image_size or 32
        train_tf = transforms.Compose(maybe_norm([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ], 3))
        test_tf = transforms.Compose(maybe_norm([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ], 3))
    elif name == 'imagenet':
        size = image_size or 224
        train_tf = transforms.Compose(maybe_norm([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ], 3))
        test_tf = transforms.Compose(maybe_norm([
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ], 3))
    else:
        raise ValueError(f'Unsupported dataset: {name}')
    return train_tf, test_tf


def build_datasets(name: str, root: str, normalization: str = 'none', image_size: Optional[int] = None):
    datasets, _ = _tv()
    name = name.lower()
    train_tf, test_tf = build_transforms(name, normalization=normalization, image_size=image_size)
    if name == 'mnist':
        train_ds = datasets.MNIST(root=root, train=True, download=True, transform=train_tf)
        test_ds = datasets.MNIST(root=root, train=False, download=True, transform=test_tf)
    elif name == 'cifar10':
        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)
    elif name == 'imagenet':
        train_ds = datasets.ImageFolder(root=f'{root}/train', transform=train_tf)
        test_ds = datasets.ImageFolder(root=f'{root}/val', transform=test_tf)
    else:
        raise ValueError(f'Unsupported dataset: {name}')
    return train_ds, test_ds


def build_loaders(name: str, root: str, batch_size: int, num_workers: int = 4, normalization: str = 'none', image_size: Optional[int] = None):
    train_ds, test_ds = build_datasets(name, root, normalization=normalization, image_size=image_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def select_correctly_classified_seeds(model, dataset, device, num_seeds: int = 100):
    model.eval()
    indices = []
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device)
        pred = model(x).argmax(dim=1).item()
        if pred == int(y):
            indices.append(idx)
        if len(indices) >= num_seeds:
            break
    return Subset(dataset, indices)

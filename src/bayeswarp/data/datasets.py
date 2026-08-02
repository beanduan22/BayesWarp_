from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Subset


_DATASET_META = {
    'mnist': {'num_classes': 10, 'channels': 1, 'size': 28},
    'cifar10': {'num_classes': 10, 'channels': 3, 'size': 32},
    'imagenet': {'num_classes': 1000, 'channels': 3, 'size': 224},
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def dataset_meta(name: str) -> Dict:
    return _DATASET_META[name.lower()]


def pixel_range(name: str, normalization: str = 'none') -> Tuple[torch.Tensor, torch.Tensor]:
    channels = dataset_meta(name)['channels']
    if normalization == 'imagenet' and channels == 3:
        mean = torch.tensor(IMAGENET_MEAN).view(channels, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(channels, 1, 1)
        return (0.0 - mean) / std, (1.0 - mean) / std
    zeros = torch.zeros(channels, 1, 1)
    return zeros, zeros + 1.0


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
    _, transforms = _tv()
    name = name.lower()
    mean, std = None, None
    if normalization == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD

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


def standard_augmentation(name: str):
    _, transforms = _tv()
    name = name.lower()
    if name == 'mnist':
        return transforms.Compose([])
    if name == 'cifar10':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    if name == 'imagenet':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])
    raise ValueError(f'Unsupported dataset: {name}')


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


def build_seed_dataset(
    name: str,
    root: str,
    normalization: str = 'none',
    image_size: Optional[int] = None,
    split: str = 'train',
):
    datasets, _ = _tv()
    name = name.lower()
    if split not in ('train', 'val'):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    _, test_tf = build_transforms(name, normalization=normalization, image_size=image_size)
    if name == 'mnist':
        return datasets.MNIST(root=root, train=(split == 'train'), download=True, transform=test_tf)
    if name == 'cifar10':
        return datasets.CIFAR10(root=root, train=(split == 'train'), download=True, transform=test_tf)
    if name == 'imagenet':
        return datasets.ImageFolder(root=f'{root}/{split}', transform=test_tf)
    raise ValueError(f'Unsupported dataset: {name}')


@torch.no_grad()
def select_correctly_classified_seeds(model, dataset, device, num_seeds: int = 100, seed: int = 0, skip: int = 0):
    model.eval()
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(dataset), generator=generator).tolist()
    indices = []
    matched = 0
    for idx in order:
        x, y = dataset[idx]
        pred = int(model(x.unsqueeze(0).to(device)).argmax(dim=1).item())
        if pred != int(y):
            continue
        matched += 1
        if matched <= skip:
            continue
        indices.append(idx)
        if len(indices) >= num_seeds:
            break
    return Subset(dataset, indices)

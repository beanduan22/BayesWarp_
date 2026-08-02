from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from .lenet import LeNet4, LeNet5


def _tv_models():
    try:
        from torchvision import models
    except Exception as e:
        raise ImportError(
            'torchvision is required for VGG/ResNet/EfficientNet backbones. '
            'Install a torchvision build compatible with your torch version.'
        ) from e
    return models


def _fit_head(head: nn.Linear, num_classes: int) -> nn.Linear:
    if head.out_features == num_classes:
        return head
    return nn.Linear(head.in_features, num_classes)


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == 'lenet4':
        return LeNet4(num_classes=num_classes)
    if name == 'lenet5':
        return LeNet5(num_classes=num_classes)

    models = _tv_models()
    if name == 'vgg16':
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier[-1] = _fit_head(model.classifier[-1], num_classes)
        return model
    if name == 'vgg19':
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg19(weights=weights)
        model.classifier[-1] = _fit_head(model.classifier[-1], num_classes)
        return model
    if name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = _fit_head(model.fc, num_classes)
        return model
    if name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = _fit_head(model.fc, num_classes)
        return model
    if name in ('efficientnet_b0', 'efficientnetb0'):
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = _fit_head(model.classifier[-1], num_classes)
        return model
    raise ValueError(f'Unsupported model: {name}')


def load_checkpoint(model: nn.Module, path: Optional[str], device: torch.device, pretrained: bool = True) -> nn.Module:
    if path is None:
        if not pretrained:
            raise ValueError(
                'checkpoint is null and pretrained is false: the subject model would '
                'be randomly initialised. Set one of them.'
            )
        return model
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    return model


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    candidates = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            candidates.append(module)
    if not candidates:
        raise ValueError('No convolutional layer found for Grad-CAM.')
    return candidates[-1]


def find_classifier_head(model: nn.Module) -> nn.Linear:
    candidates = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            candidates.append(module)
    if not candidates:
        raise ValueError('No linear classifier head found.')
    return candidates[-1]

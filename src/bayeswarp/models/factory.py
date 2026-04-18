from __future__ import annotations
import torch.nn as nn
from .lenet import LeNet4, LeNet5


def _tv_models():
    try:
        from torchvision import models
    except Exception as e:
        raise ImportError(
            'torchvision is required for VGG/ResNet backbones. '
            'Install a torchvision build compatible with your torch version.'
        ) from e
    return models


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
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == 'vgg19':
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg19(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f'Unsupported model: {name}')


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    candidates = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            candidates.append(module)
    if not candidates:
        raise ValueError('No convolutional layer found for Grad-CAM.')
    return candidates[-1]

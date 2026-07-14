"""ResNet18 для переноса обучения."""

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 4, freeze_backbone: bool = True) -> nn.Module:
    """ResNet18 с предобученными весами ImageNet и заменённой головой.

    При замороженной основе обучается только классификатор.
    Для полного дообучения используется функция unfreeze_all().
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def unfreeze_all(model: nn.Module) -> None:
    """Разморозить все параметры модели для полного дообучения (этап 2)."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    """Возвращает число обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

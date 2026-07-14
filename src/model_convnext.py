"""ConvNeXt-Tiny для переноса обучения."""

import torch.nn as nn
import timm

CONVNEXT_TINY = "convnext_tiny.fb_in22k_ft_in1k"
NUM_STAGES = 4


def build_convnext_tiny(num_classes: int = 4, freeze_backbone: bool = True) -> nn.Module:
    """ConvNeXt-Tiny с предобученными весами и заменённой головой.

    При freeze_backbone=True обучается только model.head.
    """
    model = timm.create_model(CONVNEXT_TINY, pretrained=True, num_classes=num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

    return model


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def unfreeze_stages(model: nn.Module, n_stages: int) -> None:
    """Разморозить последние n_stages стадий (1–4) и head."""
    _set_requires_grad(model, False)
    n_stages = max(1, min(n_stages, NUM_STAGES))
    for stage in model.stages[-n_stages:]:
        _set_requires_grad(stage, True)
    _set_requires_grad(model.head, True)


def unfreeze_all(model: nn.Module) -> None:
    """Разморозить все параметры модели."""
    _set_requires_grad(model, True)


def count_trainable_params(model: nn.Module) -> int:
    """Возвращает число обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

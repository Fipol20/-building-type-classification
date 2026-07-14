"""ConvNeXt-XXLarge для переноса обучения."""

import torch.nn as nn
import timm

from model_convnext import NUM_STAGES, count_trainable_params, unfreeze_all, unfreeze_stages

CONVNEXT_XXL = "convnext_xxlarge.clip_laion2b_soup_ft_in1k"


def build_convnext_xxl(num_classes: int = 4, freeze_backbone: bool = True) -> nn.Module:
    """ConvNeXt-XXLarge с предобученными весами и заменённой головой."""
    model = timm.create_model(CONVNEXT_XXL, pretrained=True, num_classes=num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

    return model

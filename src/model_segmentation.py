"""ConvNeXt-Tiny encoder + FPN-decoder для бинарной сегментации зданий."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from model_convnext import CONVNEXT_TINY, NUM_STAGES

NUM_ENCODER_STAGES = 4


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FPNDecoder(nn.Module):
    """FPN-style decoder: merge encoder stages -> 1-channel logits."""

    def __init__(self, encoder_channels: list[int], decoder_channels: int = 256) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(ch, decoder_channels, kernel_size=1) for ch in encoder_channels
        )
        self.smooth = nn.ModuleList(
            ConvBlock(decoder_channels, decoder_channels) for _ in encoder_channels
        )
        self.head = nn.Conv2d(decoder_channels, 1, kernel_size=1)

    def forward(self, features: list[torch.Tensor], out_size: tuple[int, int]) -> torch.Tensor:
        laterals = [conv(f) for conv, f in zip(self.lateral, features)]
        x = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            x = F.interpolate(x, size=laterals[i].shape[-2:], mode="bilinear", align_corners=False)
            x = laterals[i] + x
            x = self.smooth[i](x)
        x = self.head(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


class ConvNeXtSegmenter(nn.Module):
    """Бинарная сегментация: logits [B, 1, H, W]."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            CONVNEXT_TINY,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(range(NUM_ENCODER_STAGES)),
        )
        encoder_channels = self.encoder.feature_info.channels()
        self.decoder = FPNDecoder(encoder_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(features, out_size=x.shape[-2:])


def build_convnext_segmenter(pretrained: bool = True, freeze_encoder: bool = True) -> ConvNeXtSegmenter:
    """ConvNeXt-Tiny segmenter; при freeze_encoder обучается только decoder."""
    model = ConvNeXtSegmenter(pretrained=pretrained)
    if freeze_encoder:
        freeze_encoder_only(model)
    return model


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_encoder_only(model: ConvNeXtSegmenter) -> None:
    """Заморозить encoder, оставить обучаемым decoder."""
    _set_requires_grad(model, False)
    _set_requires_grad(model.decoder, True)


def _encoder_stage_modules(encoder: nn.Module) -> list[nn.Module]:
    """Стадии ConvNeXt: stages_0..3 (features_only) или .stages (classifier)."""
    if hasattr(encoder, "stages_0"):
        return [getattr(encoder, f"stages_{i}") for i in range(NUM_ENCODER_STAGES)]
    if hasattr(encoder, "stages"):
        return list(encoder.stages)
    raise AttributeError(f"Не найдены стадии encoder в {type(encoder).__name__}")


def unfreeze_encoder_stages(model: ConvNeXtSegmenter, n_stages: int) -> None:
    """Разморозить последние n_stages encoder + decoder."""
    _set_requires_grad(model, False)
    n_stages = max(1, min(n_stages, NUM_ENCODER_STAGES))
    for stage in _encoder_stage_modules(model.encoder)[-n_stages:]:
        _set_requires_grad(stage, True)
    _set_requires_grad(model.decoder, True)


def unfreeze_all(model: ConvNeXtSegmenter) -> None:
    _set_requires_grad(model, True)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

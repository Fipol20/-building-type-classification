"""Карта застройки по районам: скользящее окно + усреднение вероятностей.

Разбивает большое изображение на перекрывающиеся патчи, классифицирует
каждый обученной моделью (4 класса zone, см. dataset.CLASSES) и усредняет
softmax-вероятности по пикселям, чтобы получить сплошную карту без «плиток».
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import get_transforms
from utils import IMAGE_SIZE, ZONE_COLORS


def _window_starts(size: int, patch: int, stride: int) -> list[int]:
    """Позиции окон вдоль одной оси, гарантированно покрывающие весь [0, size)."""
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


@torch.no_grad()
def predict_zone_prob_map(
    image: Image.Image,
    model: torch.nn.Module,
    classes: list[str],
    patch_size: int = IMAGE_SIZE,
    stride: int | None = None,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Усреднённые softmax-вероятности zone по пикселям, shape [H, W, n_classes]."""
    stride = stride or patch_size // 2
    model = model.to(device).eval()
    transform = get_transforms(image_size=patch_size)

    image = image.convert("RGB")
    w, h = image.size
    xs = _window_starts(w, patch_size, stride)
    ys = _window_starts(h, patch_size, stride)

    n_classes = len(classes)
    prob_sum = np.zeros((h, w, n_classes), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    windows = [(x, y) for y in ys for x in xs]
    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start : batch_start + batch_size]
        tensors = [
            transform(image.crop((x, y, x + patch_size, y + patch_size))) for x, y in batch_windows
        ]
        batch_tensor = torch.stack(tensors).to(device)
        probs = F.softmax(model(batch_tensor), dim=1).cpu().numpy()

        for (x, y), p in zip(batch_windows, probs):
            prob_sum[y : y + patch_size, x : x + patch_size] += p
            count[y : y + patch_size, x : x + patch_size] += 1.0

    count = np.clip(count, a_min=1e-6, a_max=None)
    return prob_sum / count[..., None]


@torch.no_grad()
def predict_zone_map(
    image: Image.Image,
    model: torch.nn.Module,
    classes: list[str],
    patch_size: int = IMAGE_SIZE,
    stride: int | None = None,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Скользящим окном классифицирует патчи и усредняет softmax по пикселям.

    Returns:
        int-массив [H, W] с индексами классов (порядок соответствует `classes`).
    """
    avg_probs = predict_zone_prob_map(
        image, model, classes, patch_size=patch_size, stride=stride, batch_size=batch_size, device=device
    )
    return avg_probs.argmax(axis=2).astype(np.int32)


def colorize_zone_map(
    zone_map: np.ndarray,
    classes: list[str],
    colors: dict[str, tuple[int, int, int]] = ZONE_COLORS,
) -> Image.Image:
    """Раскрашивает int-карту классов в RGB для overlay-визуализации."""
    palette = np.array([colors[c] for c in classes], dtype=np.uint8)
    rgb = palette[zone_map]
    return Image.fromarray(rgb, mode="RGB")


def overlay_zone_map(
    image: Image.Image,
    zone_map: np.ndarray,
    classes: list[str],
    alpha: float = 0.45,
    colors: dict[str, tuple[int, int, int]] = ZONE_COLORS,
) -> Image.Image:
    """Накладывает цветную карту застройки на исходное изображение."""
    color_img = colorize_zone_map(zone_map, classes, colors)
    base = image.convert("RGB")
    if base.size != color_img.size:
        color_img = color_img.resize(base.size, Image.NEAREST)
    return Image.blend(base, color_img, alpha)


def zone_probs_at(
    zone_prob_map: np.ndarray,
    x: float,
    y: float,
) -> np.ndarray | None:
    """Вектор вероятностей zone в точке (x, y), shape [n_classes]."""
    h, w, _ = zone_prob_map.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return None
    return zone_prob_map[yi, xi]


def zone_class_at(zone_map: np.ndarray, classes: list[str], x: float, y: float) -> str | None:
    """Класс зоны в точке (x, y) изображения (например, центроид здания)."""
    h, w = zone_map.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return None
    return classes[int(zone_map[yi, xi])]

"""INRIA-сегментация на целом тайле: sliding window и mask → instances."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import ndimage
from torchvision import transforms

from building_masks import Building
from utils import IMAGENET_MEAN, IMAGENET_STD, INRIA_PATCH_SIZE, INRIA_PATCH_STRIDE


def _window_starts(size: int, patch: int, stride: int) -> list[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _normalize_patch_tensor(image_t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image_t - mean) / std


@torch.no_grad()
def predict_building_mask_sliding(
    image: Image.Image,
    model: nn.Module,
    patch_size: int = INRIA_PATCH_SIZE,
    stride: int = INRIA_PATCH_STRIDE,
    batch_size: int = 8,
    threshold: float = 0.5,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window: усреднение sigmoid-вероятностей → prob_map и бинарная mask."""
    image = image.convert("RGB")
    w, h = image.size
    xs = _window_starts(w, patch_size, stride)
    ys = _window_starts(h, patch_size, stride)

    model = model.to(device).eval()
    prob_sum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    to_tensor = transforms.ToTensor()

    windows = [(x, y) for y in ys for x in xs]
    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start : batch_start + batch_size]
        tensors = []
        for x, y in batch_windows:
            crop = image.crop((x, y, x + patch_size, y + patch_size))
            tensors.append(_normalize_patch_tensor(to_tensor(crop)))
        batch_tensor = torch.stack(tensors).to(device)
        logits = model(batch_tensor)
        probs = torch.sigmoid(logits)[:, 0].cpu().numpy()

        for (x, y), prob in zip(batch_windows, probs):
            prob_sum[y : y + patch_size, x : x + patch_size] += prob
            count[y : y + patch_size, x : x + patch_size] += 1.0

    count = np.clip(count, 1e-6, None)
    prob_map = prob_sum / count
    mask = (prob_map >= threshold).astype(np.uint8)
    return prob_map, mask


def mask_to_buildings(
    mask: np.ndarray,
    min_area: int = 32,
    image_id: int = -1,
) -> list[Building]:
    """Connected components бинарной маски → список Building с bbox и полигоном-контуром."""
    labeled, n_components = ndimage.label(mask.astype(np.uint8))
    buildings: list[Building] = []

    for comp_id in range(1, n_components + 1):
        component = labeled == comp_id
        area = int(component.sum())
        if area < min_area:
            continue

        slices = ndimage.find_objects(component.astype(np.uint8))[0]
        if slices is None:
            continue
        row_slice, col_slice = slices
        x = float(col_slice.start)
        y = float(row_slice.start)
        w = float(col_slice.stop - col_slice.start)
        h = float(row_slice.stop - row_slice.start)
        polygon = [x, y, x + w, y, x + w, y + h, x, y + h]

        buildings.append(
            Building(
                image_id=image_id,
                ann_id=comp_id,
                bbox=(x, y, w, h),
                polygons=[polygon],
                gt_class="unknown",
            )
        )

    return buildings


def rasterize_buildings_mask(buildings: list[Building], size: tuple[int, int]) -> np.ndarray:
    """Объединённая бинарная маска GT-полигонов (w, h)."""
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in buildings:
        x, y, bw, bh = b.bbox
        xi, yi = int(x), int(y)
        x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
        if x2 > xi and y2 > yi:
            mask[yi:y2, xi:x2] = 1
        for poly in b.polygons:
            if len(poly) >= 6:
                xs = [int(poly[i]) for i in range(0, len(poly), 2)]
                ys = [int(poly[i + 1]) for i in range(0, len(poly), 2)]
                y0, y1 = max(0, min(ys)), min(h, max(ys) + 1)
                x0, x1 = max(0, min(xs)), min(w, max(xs) + 1)
                if y1 > y0 and x1 > x0:
                    mask[y0:y1, x0:x1] = 1
    return mask


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return float(inter / union) if union > 0 else 1.0 if inter == 0 else 0.0

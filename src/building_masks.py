"""Маски и классификация отдельных зданий по разметке UBC.

Для тайла UBC (600x600 TIF) читает полигоны зданий из COCO-аннотаций
`data/raw/ubc/annotations/use_coarse_{split}.json`, вырезает crop по
bbox+padding (как при подготовке train-данных, см. scripts/download_extra_datasets.py)
и классифицирует его обученной моделью (residential/commercial/industrial).

Если модель не передана — используется GT-класс из разметки UBC (это позволяет
демонстрировать пайплайн до обучения building-классификатора).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from dataset import apply_building_mask, get_transforms
from utils import BUILDING_COLORS, BUILDING_MASK_BG_GRAY, IMAGE_SIZE, UBC_CLASSES, UBC_RAW_DIR

CROP_PADDING = 16  # совпадает с extract_ubc_crops в scripts/download_extra_datasets.py
MIN_CROP_SIZE = 32


@dataclass
class Building:
    """Одно здание на тайле: полигон, bbox, GT-класс и (опционально) предсказание."""

    image_id: int
    ann_id: int
    bbox: tuple[float, float, float, float]  # x, y, w, h в пикселях тайла
    polygons: list[list[float]]  # COCO segmentation: список полигонов [x1,y1,x2,y2,...]
    gt_class: str
    pred_class: str | None = None
    confidence: float | None = None
    used_gt_fallback: bool = field(default=False)
    final_class: str | None = None  # заполняется merge_maps.merge_zone_and_buildings
    merge_note: str | None = None

    @property
    def centroid(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2

    @property
    def final_source_class(self) -> str:
        """Класс, который используется дальше (предсказанный, либо GT как fallback)."""
        if self.pred_class is not None:
            return self.pred_class
        if self.gt_class != "unknown":
            return self.gt_class
        raise ValueError(f"Building {self.ann_id}: нет pred_class и GT неизвестен")


@lru_cache(maxsize=4)
def _load_annotations(json_path_str: str) -> dict:
    json_path = Path(json_path_str)
    return json.loads(json_path.read_text(encoding="utf-8"))


def get_buildings_for_tile(
    tile_filename: str,
    split: str,
    ubc_raw_dir: Path = UBC_RAW_DIR,
    classes: list[str] = UBC_CLASSES,
) -> list[Building]:
    """Здания (residential/commercial/industrial) на конкретном тайле UBC.

    Args:
        tile_filename: имя файла тайла, как в COCO ("...RGB.tif").
        split: "train" или "val" (какой use_coarse_*.json читать).
    """
    json_path = ubc_raw_dir / "annotations" / f"use_coarse_{split}.json"
    data = _load_annotations(str(json_path))

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    target_cat_ids = {cid for cid, name in cat_id_to_name.items() if name in classes}

    image_id = None
    for img in data["images"]:
        if img["file_name"] == tile_filename:
            image_id = img["id"]
            break
    if image_id is None:
        return []

    buildings = []
    for ann in data["annotations"]:
        if ann["image_id"] != image_id or ann["category_id"] not in target_cat_ids:
            continue
        buildings.append(
            Building(
                image_id=image_id,
                ann_id=ann["id"],
                bbox=tuple(ann["bbox"]),
                polygons=ann.get("segmentation") or [],
                gt_class=cat_id_to_name[ann["category_id"]],
            )
        )
    return buildings


def _crop_box(img_size: tuple[int, int], bbox: tuple[float, float, float, float], padding: int) -> tuple[int, int, int, int] | None:
    img_w, img_h = img_size
    x, y, w, h = bbox
    left = max(0, int(x) - padding)
    top = max(0, int(y) - padding)
    right = min(img_w, int(x + w) + padding)
    bottom = min(img_h, int(y + h) + padding)
    if right - left < MIN_CROP_SIZE or bottom - top < MIN_CROP_SIZE:
        return None
    return left, top, right, bottom


@torch.no_grad()
def classify_buildings(
    image: Image.Image,
    buildings: list[Building],
    model: torch.nn.Module | None,
    classes: list[str] = UBC_CLASSES,
    image_size: int = IMAGE_SIZE,
    batch_size: int = 32,
    device: str = "cpu",
    use_mask: bool = True,
    confidence_threshold: float = 0.0,
) -> list[Building]:
    """Заполняет pred_class/confidence у зданий. Без модели — fallback на GT."""
    if model is None:
        for b in buildings:
            b.pred_class = b.gt_class
            b.used_gt_fallback = True
        return buildings

    image = image.convert("RGB")
    model = model.to(device).eval()
    transform = get_transforms(image_size=image_size)

    crops, valid_buildings = [], []
    for b in buildings:
        box = _crop_box(image.size, b.bbox, CROP_PADDING)
        if box is None:
            b.pred_class = b.gt_class
            b.used_gt_fallback = True
            continue
        left, top, right, bottom = box
        crop = image.crop(box)
        if use_mask and b.polygons:
            mask = Image.new("L", (right - left, bottom - top), 0)
            draw = ImageDraw.Draw(mask)
            for poly in b.polygons:
                pts = [(poly[i] - left, poly[i + 1] - top) for i in range(0, len(poly), 2)]
                if len(pts) >= 2:
                    draw.polygon(pts, fill=255)
            crop = apply_building_mask(crop, mask, bg_gray=BUILDING_MASK_BG_GRAY)
        crops.append(transform(crop))
        valid_buildings.append(b)

    for start in range(0, len(crops), batch_size):
        batch = torch.stack(crops[start : start + batch_size]).to(device)
        probs = F.softmax(model(batch), dim=1).cpu()
        confidences, indices = probs.max(dim=1)
        for b, idx, conf in zip(valid_buildings[start : start + batch_size], indices.tolist(), confidences.tolist()):
            b.confidence = conf
            b.used_gt_fallback = False
            if conf >= confidence_threshold:
                b.pred_class = classes[idx]
            else:
                b.pred_class = None

    return buildings


def draw_building_masks(
    image: Image.Image,
    buildings: list[Building],
    colors: dict[str, tuple[int, int, int]] = BUILDING_COLORS,
    use_predicted: bool = True,
    alpha: float = 0.55,
    outline_width: int = 2,
) -> Image.Image:
    """Полупрозрачные маски зданий, раскрашенные по классу (предсказанному или GT)."""
    base = image.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for b in buildings:
        cls = (b.pred_class if use_predicted and b.pred_class else None) or b.gt_class
        color = colors.get(cls, (200, 200, 200))
        fill = (*color, int(255 * alpha))
        for poly in b.polygons:
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            if len(pts) >= 2:
                draw.polygon(pts, fill=fill, outline=(*color, 255), width=outline_width)

    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

"""Полный пайплайн на UBC-тайлах: INRIA → NY building → zone → merge."""

from __future__ import annotations

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score

from building_masks import Building, classify_buildings, get_buildings_for_tile
from dataset import CLASSES as ZONE_CLASSES
from inria_inference import mask_iou, mask_to_buildings, predict_building_mask_sliding, rasterize_buildings_mask
from merge_maps import merge_summary, merge_zone_and_buildings
from utils import NY_BUILDING_CLASSES, UBC_RAW_DIR, UBC_SPLIT_CSV
from zone_map import overlay_zone_map, predict_zone_prob_map


@dataclass
class PipelineResult:
    tile_name: str
    split: str
    image: Image.Image
    prob_map: np.ndarray | None
    pred_mask: np.ndarray | None
    zone_map: np.ndarray | None
    buildings: list[Building]
    gt_buildings: list[Building]
    mask_iou: float | None = None


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_buildings(
    pred: list[Building],
    gt: list[Building],
    iou_threshold: float = 0.2,
) -> list[tuple[Building, Building]]:
    """Жадное сопоставление pred↔GT по IoU bbox."""
    pairs: list[tuple[Building, Building]] = []
    used_gt: set[int] = set()
    for p in sorted(pred, key=lambda b: b.bbox[2] * b.bbox[3], reverse=True):
        best_iou, best_g = 0.0, None
        for g in gt:
            if g.ann_id in used_gt:
                continue
            iou = _bbox_iou(p.bbox, g.bbox)
            if iou > best_iou:
                best_iou, best_g = iou, g
        if best_g is not None and best_iou >= iou_threshold:
            pairs.append((p, best_g))
            used_gt.add(best_g.ann_id)
    return pairs


def resolve_ubc_raw_split(
    tile_name: str,
    ubc_raw_dir: Path = UBC_RAW_DIR,
) -> str:
    """Каталог raw (`train`/`val`), где лежит *_RGB.tif (и соответствующий use_coarse_*.json)."""
    for split in ("train", "val"):
        if (ubc_raw_dir / split / tile_name).exists():
            return split
    raise FileNotFoundError(
        f"Тайл {tile_name!r} не найден ни в {ubc_raw_dir / 'train'}, ни в {ubc_raw_dir / 'val'}"
    )


def ubc_tile_names_for_eval_split(split: str, ubc_split_csv: Path = UBC_SPLIT_CSV) -> list[str]:
    """Имена *_RGB.tif для val/test по image_id из processed_ubc/split.csv."""
    if not ubc_split_csv.exists():
        return []

    df = pd.read_csv(ubc_split_csv)
    subset = df[df["split"] == split]
    if subset.empty:
        return []

    image_ids: set[int] = set()
    for path in subset["path"]:
        stem = Path(path).stem
        if stem.endswith("_mask"):
            stem = stem[:-5]
        parts = stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            image_ids.add(int(parts[1]))

    id_to_name: dict[int, str] = {}
    for json_path in sorted((UBC_RAW_DIR / "annotations").glob("use_coarse_*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for img in data.get("images", []):
            if img["id"] in image_ids:
                id_to_name[img["id"]] = img["file_name"]

    return sorted({id_to_name[i] for i in image_ids if i in id_to_name})


@torch.no_grad()
def run_pipeline_on_tile(
    tile_name: str,
    split: str,
    zone_model: torch.nn.Module | None,
    segmenter_model: torch.nn.Module,
    building_model: torch.nn.Module | None,
    device: str = "cpu",
    min_component_area: int = 32,
    mask_threshold: float = 0.5,
    ubc_raw_dir: Path = UBC_RAW_DIR,
    pipeline_params: "PipelineParams | None" = None,
) -> PipelineResult:
    if pipeline_params is not None:
        min_component_area = pipeline_params.min_component_area
        mask_threshold = pipeline_params.mask_threshold

    image_path = ubc_raw_dir / split / tile_name
    image = Image.open(image_path).convert("RGB")

    prob_map, pred_mask = predict_building_mask_sliding(
        image,
        segmenter_model,
        threshold=mask_threshold,
        device=device,
    )
    buildings = mask_to_buildings(pred_mask, min_area=min_component_area)
    conf_threshold = pipeline_params.class_confidence_threshold if pipeline_params else 0.0
    classify_buildings(
        image,
        buildings,
        model=building_model,
        classes=NY_BUILDING_CLASSES,
        device=device,
        use_mask=True,
        confidence_threshold=conf_threshold,
    )

    zone_map = None
    zone_prob_map = None
    if zone_model is not None:
        zone_prob_map = predict_zone_prob_map(image, zone_model, ZONE_CLASSES, device=device)
        zone_map = zone_prob_map.argmax(axis=2).astype(np.int32)

    zone_min_prob = pipeline_params.zone_residential_min_prob if pipeline_params else 0.0
    merge_zone_and_buildings(
        buildings,
        zone_map,
        ZONE_CLASSES,
        zone_prob_map=zone_prob_map,
        zone_residential_min_prob=zone_min_prob,
    )

    gt_buildings = get_buildings_for_tile(tile_name, split=split, ubc_raw_dir=ubc_raw_dir)
    gt_mask = rasterize_buildings_mask(gt_buildings, image.size)
    miou = mask_iou(pred_mask, gt_mask)

    return PipelineResult(
        tile_name=tile_name,
        split=split,
        image=image,
        prob_map=prob_map,
        pred_mask=pred_mask,
        zone_map=zone_map,
        buildings=buildings,
        gt_buildings=gt_buildings,
        mask_iou=miou,
    )


def evaluate_building_classification(
    result: PipelineResult,
    iou_threshold: float = 0.2,
) -> dict[str, float]:
    pairs = match_buildings(result.buildings, result.gt_buildings, iou_threshold=iou_threshold)
    if not pairs:
        return {"matched": 0.0, "accuracy": 0.0, "macro_f1": 0.0}

    y_true = [g.gt_class for _, g in pairs]
    y_pred = [p.pred_class for p, _ in pairs if p.pred_class is not None]
    if not y_pred:
        return {"matched": float(len(pairs)), "accuracy": 0.0, "macro_f1": 0.0}
    pairs_with_pred = [(p, g) for p, g in pairs if p.pred_class is not None]
    y_true = [g.gt_class for _, g in pairs_with_pred]
    y_pred = [p.pred_class for p, _ in pairs_with_pred]
    labels = list(NY_BUILDING_CLASSES)
    return {
        "matched": float(len(pairs)),
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def zone_overlay(image: Image.Image, zone_map: np.ndarray | None) -> Image.Image:
    if zone_map is None:
        return image
    return overlay_zone_map(image, zone_map, ZONE_CLASSES)

"""Совместная калибровка порогов пайплайна zone + find + class + merge."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
from PIL import Image

from building_masks import Building, classify_buildings, get_buildings_for_tile
from dataset import CLASSES as ZONE_CLASSES
from inria_inference import mask_iou, mask_to_buildings, predict_building_mask_sliding, rasterize_buildings_mask
from merge_maps import RESIDENTIAL_ZONE_CLASSES, merge_zone_and_buildings
from pipeline_ubc import PipelineResult, evaluate_building_classification, resolve_ubc_raw_split
from utils import NY_BUILDING_CLASSES, REPORTS_DIR, UBC_RAW_DIR
from zone_map import predict_zone_prob_map

DEFAULT_PARAMS_PATH = REPORTS_DIR / "pipeline_calibrated_params.json"

GridMode = Literal["full", "fast"]

VariantKey = tuple[float, int]


@dataclass
class PipelineParams:
    mask_threshold: float = 0.45
    min_component_area: int = 32
    class_confidence_threshold: float = 0.0
    zone_residential_min_prob: float = 0.0
    match_iou: float = 0.2
    score_weight_mask: float = 0.4
    score_weight_class: float = 0.5
    score_weight_merge: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PipelineParams:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def save_json(self, path: Path = DEFAULT_PARAMS_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: Path = DEFAULT_PARAMS_PATH) -> PipelineParams | None:
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class TileCache:
    tile_name: str
    raw_split: str
    prob_map: np.ndarray
    zone_prob_map: np.ndarray
    gt_buildings: list[Building]
    gt_mask: np.ndarray
    image: Image.Image


@dataclass
class FindClassVariant:
    """Find + class (conf=0) для пары mask_threshold × min_component_area."""

    pred_mask: np.ndarray
    buildings: list[Building]


@dataclass
class CalibrationMetrics:
    mask_iou: float
    building_macro_f1: float
    building_acc: float
    resolved_residential_rate: float
    joint_score: float
    n_tiles: int
    n_matched: float


@dataclass
class CalibrationResult:
    best_params: PipelineParams
    best_metrics: CalibrationMetrics
    top_results: list[dict]


def _resolved_residential_rate(buildings: list[Building]) -> float:
    residential = [b for b in buildings if b.pred_class == "residential" and b.final_class is not None]
    if not residential:
        return 0.0
    resolved = sum(1 for b in residential if b.final_class in RESIDENTIAL_ZONE_CLASSES)
    return resolved / len(residential)


def _apply_confidence_threshold(buildings: list[Building], threshold: float) -> list[Building]:
    out = copy.deepcopy(buildings)
    for b in out:
        if b.confidence is not None and b.confidence < threshold:
            b.pred_class = None
            b.final_class = None
            b.merge_note = None
    return out


def run_pipeline_from_variant(
    cache: TileCache,
    variant: FindClassVariant,
    params: PipelineParams,
) -> PipelineResult:
    buildings = _apply_confidence_threshold(variant.buildings, params.class_confidence_threshold)
    zone_map = cache.zone_prob_map.argmax(axis=2).astype(np.int32)
    merge_zone_and_buildings(
        buildings,
        zone_map,
        ZONE_CLASSES,
        zone_prob_map=cache.zone_prob_map,
        zone_residential_min_prob=params.zone_residential_min_prob,
    )
    return PipelineResult(
        tile_name=cache.tile_name,
        split=cache.raw_split,
        image=cache.image,
        prob_map=cache.prob_map,
        pred_mask=variant.pred_mask,
        zone_map=zone_map,
        buildings=buildings,
        gt_buildings=cache.gt_buildings,
        mask_iou=mask_iou(variant.pred_mask, cache.gt_mask),
    )


def run_pipeline_from_cache(
    cache: TileCache,
    params: PipelineParams,
    class_model: torch.nn.Module | None = None,
    device: str = "cpu",
) -> PipelineResult:
    pred_mask = (cache.prob_map >= params.mask_threshold).astype(np.uint8)
    buildings = mask_to_buildings(pred_mask, min_area=params.min_component_area)
    classify_buildings(
        cache.image,
        buildings,
        model=class_model,
        classes=NY_BUILDING_CLASSES,
        device=device,
        use_mask=True,
        confidence_threshold=params.class_confidence_threshold,
    )
    zone_map = cache.zone_prob_map.argmax(axis=2).astype(np.int32)
    merge_zone_and_buildings(
        buildings,
        zone_map,
        ZONE_CLASSES,
        zone_prob_map=cache.zone_prob_map,
        zone_residential_min_prob=params.zone_residential_min_prob,
    )
    return PipelineResult(
        tile_name=cache.tile_name,
        split=cache.raw_split,
        image=cache.image,
        prob_map=cache.prob_map,
        pred_mask=pred_mask,
        zone_map=zone_map,
        buildings=buildings,
        gt_buildings=cache.gt_buildings,
        mask_iou=mask_iou(pred_mask, cache.gt_mask),
    )


@torch.no_grad()
def cache_tile_outputs(
    tile_name: str,
    zone_model: torch.nn.Module,
    find_model: torch.nn.Module,
    device: str = "cpu",
    raw_split: str | None = None,
    seg_batch_size: int | None = None,
) -> TileCache:
    """Один проход тяжёлых моделей: prob_map INRIA + zone_prob_map + GT."""
    if raw_split is None:
        raw_split = resolve_ubc_raw_split(tile_name)
    image_path = UBC_RAW_DIR / raw_split / tile_name
    image = Image.open(image_path).convert("RGB")

    kwargs = {"device": device}
    if seg_batch_size is not None:
        kwargs["batch_size"] = seg_batch_size

    prob_map, _ = predict_building_mask_sliding(image, find_model, threshold=0.5, **kwargs)
    zone_prob_map = predict_zone_prob_map(image, zone_model, ZONE_CLASSES, device=device)
    gt_buildings = get_buildings_for_tile(tile_name, split=raw_split)
    gt_mask = rasterize_buildings_mask(gt_buildings, image.size)

    return TileCache(
        tile_name=tile_name,
        raw_split=raw_split,
        prob_map=prob_map,
        zone_prob_map=zone_prob_map,
        gt_buildings=gt_buildings,
        gt_mask=gt_mask,
        image=image,
    )


def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


@torch.no_grad()
def precompute_find_class_variants(
    caches: list[TileCache],
    keys: set[VariantKey],
    class_model: torch.nn.Module | None,
    device: str = "cpu",
    class_batch_size: int | None = None,
    *,
    verbose: bool = True,
) -> dict[VariantKey, list[FindClassVariant]]:
    """Classify один раз на пару (mask_threshold, min_area); conf/zone — без модели."""
    store: dict[VariantKey, list[FindClassVariant]] = {}
    cls_kwargs: dict = {"device": device, "use_mask": True, "confidence_threshold": 0.0}
    if class_batch_size is not None:
        cls_kwargs["batch_size"] = class_batch_size

    sorted_keys = sorted(keys)
    _log(f"[calib] precompute classify: {len(sorted_keys)} пар (mask, area) × {len(caches)} тайлов", verbose)

    try:
        from tqdm.auto import tqdm as _tqdm

        key_iter: Iterable[VariantKey] = _tqdm(sorted_keys, desc="precompute find+class")
    except ImportError:
        key_iter = sorted_keys

    for mask_th, min_area in key_iter:
        _log(f"[calib]   classify mask={mask_th:.2f} area={min_area}", verbose)
        variants: list[FindClassVariant] = []
        for cache in caches:
            pred_mask = (cache.prob_map >= mask_th).astype(np.uint8)
            buildings = mask_to_buildings(pred_mask, min_area=min_area)
            classify_buildings(
                cache.image,
                buildings,
                model=class_model,
                classes=NY_BUILDING_CLASSES,
                **cls_kwargs,
            )
            variants.append(FindClassVariant(pred_mask=pred_mask, buildings=buildings))
        store[(mask_th, min_area)] = variants
    return store


def evaluate_params(
    caches: list[TileCache],
    params: PipelineParams,
    class_model: torch.nn.Module | None = None,
    device: str = "cpu",
    *,
    variant_store: dict[VariantKey, list[FindClassVariant]] | None = None,
) -> CalibrationMetrics:
    mask_ious: list[float] = []
    macro_f1s: list[float] = []
    accs: list[float] = []
    resolved_rates: list[float] = []
    total_matched = 0.0

    find_key = (params.mask_threshold, params.min_component_area)

    for i, cache in enumerate(caches):
        if variant_store is not None and find_key in variant_store:
            result = run_pipeline_from_variant(cache, variant_store[find_key][i], params)
        else:
            result = run_pipeline_from_cache(cache, params, class_model=class_model, device=device)
        mask_ious.append(result.mask_iou or 0.0)
        cls_m = evaluate_building_classification(result, iou_threshold=params.match_iou)
        if cls_m["matched"] > 0:
            macro_f1s.append(cls_m["macro_f1"])
            accs.append(cls_m["accuracy"])
        total_matched += cls_m["matched"]
        resolved_rates.append(_resolved_residential_rate(result.buildings))

    mean_mask = float(np.mean(mask_ious)) if mask_ious else 0.0
    mean_f1 = float(np.mean(macro_f1s)) if macro_f1s else 0.0
    mean_acc = float(np.mean(accs)) if accs else 0.0
    mean_resolved = float(np.mean(resolved_rates)) if resolved_rates else 0.0
    joint = (
        params.score_weight_mask * mean_mask
        + params.score_weight_class * mean_f1
        + params.score_weight_merge * mean_resolved
    )
    return CalibrationMetrics(
        mask_iou=mean_mask,
        building_macro_f1=mean_f1,
        building_acc=mean_acc,
        resolved_residential_rate=mean_resolved,
        joint_score=joint,
        n_tiles=len(caches),
        n_matched=total_matched,
    )


def _param_grid(coarse: bool = True, grid_mode: GridMode = "full") -> dict[str, list]:
    if grid_mode in ("fast", "colab"):  # colab — устаревший alias
        if coarse:
            return {
                "mask_threshold": [0.40, 0.45, 0.50],
                "min_component_area": [24, 32, 48],
                "class_confidence_threshold": [0.0, 0.35, 0.5, 0.65],
                "zone_residential_min_prob": [0.0, 0.35, 0.5, 0.65],
            }
        return {
            "mask_threshold": [0.42, 0.44, 0.46, 0.48],
            "min_component_area": [28, 32, 36, 40],
            "class_confidence_threshold": [0.4, 0.45, 0.5, 0.55],
            "zone_residential_min_prob": [0.3, 0.4, 0.45, 0.5],
        }

    if coarse:
        return {
            "mask_threshold": [0.35, 0.40, 0.45, 0.50, 0.55],
            "min_component_area": [16, 32, 48, 64],
            "class_confidence_threshold": [0.0, 0.3, 0.5, 0.6, 0.7],
            "zone_residential_min_prob": [0.0, 0.35, 0.5, 0.65],
        }
    return {
        "mask_threshold": [0.40, 0.42, 0.44, 0.46, 0.48],
        "min_component_area": [24, 32, 40],
        "class_confidence_threshold": [0.4, 0.5, 0.55, 0.6],
        "zone_residential_min_prob": [0.3, 0.4, 0.5, 0.55],
    }


def _refine_grid(best: PipelineParams) -> dict[str, list]:
    def _around(val: float, deltas: list[float], clamp_min: float = 0.0, clamp_max: float = 1.0) -> list[float]:
        return sorted({max(clamp_min, min(clamp_max, val + d)) for d in deltas})

    return {
        "mask_threshold": _around(best.mask_threshold, [-0.05, -0.02, 0.0, 0.02, 0.05]),
        "min_component_area": sorted({max(8, best.min_component_area + d) for d in [-16, -8, 0, 8, 16]}),
        "class_confidence_threshold": _around(best.class_confidence_threshold, [-0.1, -0.05, 0.0, 0.05, 0.1]),
        "zone_residential_min_prob": _around(best.zone_residential_min_prob, [-0.15, -0.05, 0.0, 0.05, 0.15]),
    }


def _unique_find_keys(grid: dict[str, list]) -> set[VariantKey]:
    return {(m, a) for m in grid["mask_threshold"] for a in grid["min_component_area"]}


def _search_grid(
    caches: list[TileCache],
    grid: dict[str, list],
    base: PipelineParams,
    class_model: torch.nn.Module | None,
    device: str,
    *,
    variant_store: dict[VariantKey, list[FindClassVariant]] | None = None,
    stage: str = "grid",
    verbose: bool = True,
) -> tuple[PipelineParams, CalibrationMetrics, list[dict]]:
    keys = list(grid.keys())
    combos = list(product(*(grid[k] for k in keys)))
    total = len(combos)
    _log(f"[calib] {stage}: перебор {total} комбинаций ({', '.join(keys)})", verbose)

    best_params = copy.deepcopy(base)
    best_metrics = evaluate_params(
        caches, best_params, class_model, device, variant_store=variant_store
    )
    rows: list[dict] = []

    for i, combo in enumerate(combos, start=1):
        p = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            setattr(p, k, v)
        m = evaluate_params(caches, p, class_model, device, variant_store=variant_store)
        rows.append({**p.to_dict(), **asdict(m)})
        if m.joint_score > best_metrics.joint_score:
            best_metrics = m
            best_params = p
            _log(
                f"[calib] {stage} [{i}/{total}] NEW BEST joint={m.joint_score:.4f} | "
                f"mask={p.mask_threshold:.2f} area={p.min_component_area} "
                f"conf={p.class_confidence_threshold:.2f} zone={p.zone_residential_min_prob:.2f} | "
                f"IoU={m.mask_iou:.3f} F1={m.building_macro_f1:.3f}",
                verbose,
            )
        elif verbose and (i == 1 or i == total or i % max(1, total // 10) == 0):
            _log(
                f"[calib] {stage} [{i}/{total}] joint={m.joint_score:.4f} (best={best_metrics.joint_score:.4f})",
                verbose,
            )

    rows.sort(key=lambda r: r["joint_score"], reverse=True)
    _log(
        f"[calib] {stage} готово: best joint={best_metrics.joint_score:.4f} | "
        f"mask={best_params.mask_threshold:.2f} conf={best_params.class_confidence_threshold:.2f} "
        f"zone={best_params.zone_residential_min_prob:.2f}",
        verbose,
    )
    return best_params, best_metrics, rows


def calibrate(
    caches: list[TileCache],
    base_params: PipelineParams | None = None,
    class_model: torch.nn.Module | None = None,
    device: str = "cpu",
    refine: bool = True,
    *,
    fast: bool = True,
    grid_mode: GridMode = "fast",
    class_batch_size: int | None = None,
    verbose: bool = True,
) -> CalibrationResult:
    """Coarse grid → refine. fast=True: classify только на уникальных (mask, area) парах."""
    base = base_params or PipelineParams()
    coarse_grid = _param_grid(coarse=True, grid_mode=grid_mode)
    variant_store: dict[VariantKey, list[FindClassVariant]] | None = None

    _log(f"[calib] старт: {len(caches)} тайлов, grid_mode={grid_mode}, fast={fast}, refine={refine}", verbose)

    if fast:
        find_keys = _unique_find_keys(coarse_grid)
        variant_store = precompute_find_class_variants(
            caches,
            find_keys,
            class_model,
            device,
            class_batch_size=class_batch_size,
            verbose=verbose,
        )

    best_p, best_m, coarse_rows = _search_grid(
        caches,
        coarse_grid,
        base,
        class_model,
        device,
        variant_store=variant_store,
        stage="coarse",
        verbose=verbose,
    )

    all_rows = coarse_rows
    if refine:
        fine_grid = _refine_grid(best_p)
        if fast:
            new_keys = _unique_find_keys(fine_grid) - set(variant_store or {})
            if new_keys:
                _log(f"[calib] refine: +{len(new_keys)} новых пар для classify", verbose)
                extra = precompute_find_class_variants(
                    caches,
                    new_keys,
                    class_model,
                    device,
                    class_batch_size=class_batch_size,
                    verbose=verbose,
                )
                variant_store = {**(variant_store or {}), **extra}
        best_p, best_m, fine_rows = _search_grid(
            caches,
            fine_grid,
            best_p,
            class_model,
            device,
            variant_store=variant_store,
            stage="refine",
            verbose=verbose,
        )
        all_rows = coarse_rows + fine_rows
        all_rows.sort(key=lambda r: r["joint_score"], reverse=True)

    _log("[calib] === итог лучшие параметры ===", verbose)
    for k, v in best_p.to_dict().items():
        _log(f"[calib]   {k}: {v}", verbose)
    _log(
        f"[calib]   joint={best_m.joint_score:.4f} mask_iou={best_m.mask_iou:.4f} "
        f"F1={best_m.building_macro_f1:.4f}",
        verbose,
    )

    return CalibrationResult(
        best_params=best_p,
        best_metrics=best_m,
        top_results=all_rows[:10],
    )

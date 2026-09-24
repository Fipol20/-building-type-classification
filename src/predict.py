"""Inference: 4-этапный пайплайн (ноутбук 07) и классификация одного кропа (практика_ml.md)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from building_masks import Building, classify_buildings, draw_building_masks
from dataset import CLASSES as ZONE_CLASSES, get_transforms
from inria_inference import mask_to_buildings, predict_building_mask_sliding
from merge_maps import draw_merged_masks, merge_summary, merge_zone_and_buildings
from model_convnext import build_convnext_tiny
from model_segmentation import build_convnext_segmenter
from pipeline_calibrate import DEFAULT_PARAMS_PATH, PipelineParams
from pipeline_ubc import run_pipeline_on_tile, zone_overlay
from utils import IMAGE_SIZE, MODELS_DIR, NY_BUILDING_CLASSES, REPORTS_DIR, UBC_RAW_DIR
from zone_map import predict_zone_prob_map


@dataclass
class PipelineBundle:
    """Загруженные модели и пороги для inference."""

    device: torch.device
    zone_model: torch.nn.Module
    find_model: torch.nn.Module
    class_model: torch.nn.Module
    params: PipelineParams
    zone_ckpt: Path
    find_ckpt: Path
    class_ckpt: Path


@dataclass
class TilePipelineResult:
    """Результат 4-этапного пайплайна на одном изображении."""

    image: Image.Image
    zone_map: np.ndarray
    zone_prob_map: np.ndarray
    pred_mask: np.ndarray
    buildings_find: list[Building]
    buildings_class: list[Building]
    buildings_merge: list[Building]
    zone_probs_mean: dict[str, float] = field(default_factory=dict)
    merge_distribution: dict[str, float] = field(default_factory=dict)
    dominant_merged_class: str | None = None


def _pick_checkpoint(*names: str) -> Path:
    for name in names:
        path = MODELS_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Нет ни одного из чекпоинтов: {', '.join(names)}")


def _load_state(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))


def load_pipeline(device: torch.device | None = None) -> PipelineBundle:
    """Загружает 3 модели и калиброванные пороги (как в ноутбуке 07)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    zone_ckpt = _pick_checkpoint("convnext_best.pth")
    find_ckpt = _pick_checkpoint("inria_building_ubc.pth", "inria_building_segmenter.pth")
    class_ckpt = _pick_checkpoint("ny_building_ubc.pth", "ny_building_classifier.pth", "best_model.pth")

    zone_model = build_convnext_tiny(num_classes=len(ZONE_CLASSES), freeze_backbone=False)
    find_model = build_convnext_segmenter(pretrained=False, freeze_encoder=False)
    class_model = build_convnext_tiny(num_classes=len(NY_BUILDING_CLASSES), freeze_backbone=False)

    _load_state(zone_model, zone_ckpt, device)
    _load_state(find_model, find_ckpt, device)
    _load_state(class_model, class_ckpt, device)

    zone_model.to(device).eval()
    find_model.to(device).eval()
    class_model.to(device).eval()

    params = PipelineParams.load_json(DEFAULT_PARAMS_PATH) or PipelineParams()

    return PipelineBundle(
        device=device,
        zone_model=zone_model,
        find_model=find_model,
        class_model=class_model,
        params=params,
        zone_ckpt=zone_ckpt,
        find_ckpt=find_ckpt,
        class_ckpt=class_ckpt,
    )


def mask_overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int] = (255, 80, 80)) -> Image.Image:
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask.astype(bool)] = (*color, 140)
    return Image.alpha_composite(image.convert("RGBA"), Image.fromarray(overlay, "RGBA")).convert("RGB")


def _mean_zone_probs(zone_prob_map: np.ndarray) -> dict[str, float]:
    mean = zone_prob_map.mean(axis=(0, 1))
    return {cls: float(p) for cls, p in zip(ZONE_CLASSES, mean)}


def _merge_fractions(buildings: list[Building]) -> dict[str, float]:
    counts = merge_summary(buildings)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {cls: cnt / total for cls, cnt in counts.items()}


@torch.no_grad()
def predict_tile_pipeline(image: Image.Image, bundle: PipelineBundle | None = None) -> TilePipelineResult:
    """4 этапа на спутниковом снимке: zone → find → class → merge."""
    bundle = bundle or load_pipeline()
    device = str(bundle.device)
    params = bundle.params
    image = image.convert("RGB")

    zone_prob_map = predict_zone_prob_map(image, bundle.zone_model, ZONE_CLASSES, device=device)
    zone_map = zone_prob_map.argmax(axis=2).astype(np.int32)

    _, pred_mask = predict_building_mask_sliding(
        image,
        bundle.find_model,
        threshold=params.mask_threshold,
        device=device,
    )
    buildings = mask_to_buildings(pred_mask, min_area=params.min_component_area)
    classify_buildings(
        image,
        buildings,
        model=bundle.class_model,
        classes=NY_BUILDING_CLASSES,
        device=device,
        use_mask=True,
        confidence_threshold=params.class_confidence_threshold,
    )
    merge_zone_and_buildings(
        buildings,
        zone_map,
        ZONE_CLASSES,
        zone_prob_map=zone_prob_map,
        zone_residential_min_prob=params.zone_residential_min_prob,
    )

    buildings_find = mask_to_buildings(pred_mask, min_area=params.min_component_area)
    buildings_class = [b for b in buildings if b.pred_class is not None]
    buildings_merge = [b for b in buildings if b.final_class is not None]

    merge_fr = _merge_fractions(buildings_merge)
    dominant = max(merge_fr, key=merge_fr.get) if merge_fr else None

    return TilePipelineResult(
        image=image,
        zone_map=zone_map,
        zone_prob_map=zone_prob_map,
        pred_mask=pred_mask,
        buildings_find=buildings_find,
        buildings_class=buildings_class,
        buildings_merge=buildings_merge,
        zone_probs_mean=_mean_zone_probs(zone_prob_map),
        merge_distribution=merge_fr,
        dominant_merged_class=dominant,
    )


@torch.no_grad()
def predict_building_crop(image: Image.Image, bundle: PipelineBundle | None = None) -> dict[str, Any]:
    """Классификация одного кропа здания (требование практика_ml.md)."""
    bundle = bundle or load_pipeline()
    transform = get_transforms(image_size=IMAGE_SIZE)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(bundle.device)
    probs = F.softmax(bundle.class_model(tensor), dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return {
        "predicted_class": NY_BUILDING_CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {cls: float(p) for cls, p in zip(NY_BUILDING_CLASSES, probs)},
    }


def tile_stage_images(result: TilePipelineResult) -> dict[str, Image.Image]:
    """Визуализации 4 этапов для Streamlit."""
    image = result.image
    out: dict[str, Image.Image] = {"original": image.copy()}
    out["zone"] = zone_overlay(image, result.zone_map)
    out["find"] = mask_overlay(image, result.pred_mask)
    out["class"] = draw_building_masks(image, result.buildings_class, use_predicted=True)
    if result.buildings_merge:
        out["merge"] = draw_merged_masks(image, result.buildings_merge)
    else:
        out["merge"] = image.copy()
    return out


@torch.no_grad()
def predict_ubc_tile(tile_name: str, split: str = "val", bundle: PipelineBundle | None = None) -> TilePipelineResult:
    """Пайплайн на UBC-тайле из датасета (с GT для метрик в ноутбуке 07)."""
    bundle = bundle or load_pipeline()
    pipeline = run_pipeline_on_tile(
        tile_name,
        split=split,
        zone_model=bundle.zone_model,
        segmenter_model=bundle.find_model,
        building_model=bundle.class_model,
        device=str(bundle.device),
        pipeline_params=bundle.params,
        ubc_raw_dir=UBC_RAW_DIR,
    )
    buildings_after_find = mask_to_buildings(
        pipeline.pred_mask,
        min_area=bundle.params.min_component_area,
    )
    buildings_class = [b for b in pipeline.buildings if b.pred_class is not None]
    buildings_merge = [b for b in pipeline.buildings if b.final_class is not None]
    zone_prob_map = predict_zone_prob_map(
        pipeline.image, bundle.zone_model, ZONE_CLASSES, device=str(bundle.device)
    )
    merge_fr = _merge_fractions(buildings_merge)
    return TilePipelineResult(
        image=pipeline.image,
        zone_map=pipeline.zone_map,
        zone_prob_map=zone_prob_map,
        pred_mask=pipeline.pred_mask,
        buildings_find=buildings_after_find,
        buildings_class=buildings_class,
        buildings_merge=buildings_merge,
        zone_probs_mean=_mean_zone_probs(zone_prob_map),
        merge_distribution=merge_fr,
        dominant_merged_class=max(merge_fr, key=merge_fr.get) if merge_fr else None,
    )


def result_to_dict(result: TilePipelineResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, TilePipelineResult):
        return {
            "n_buildings_find": len(result.buildings_find),
            "n_buildings_class": len(result.buildings_class),
            "n_buildings_merge": len(result.buildings_merge),
            "dominant_merged_class": result.dominant_merged_class,
            "zone_probs_mean": result.zone_probs_mean,
            "merge_distribution": result.merge_distribution,
        }
    return result


def save_tile_result(
    result: TilePipelineResult,
    image_path: Path,
    out_dir: Path | None = None,
) -> dict[str, str]:
    """Сохраняет маски и визуализации этапов пайплайна."""
    stem = image_path.stem
    out_dir = (out_dir or REPORTS_DIR / "predictions") / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}

    mask_path = out_dir / "building_mask.png"
    Image.fromarray((result.pred_mask.astype(np.uint8) * 255), mode="L").save(mask_path)
    saved["building_mask"] = str(mask_path)

    stages = tile_stage_images(result)
    for stage_key, filename in (
        ("zone", "zone_overlay.png"),
        ("find", "find_overlay.png"),
        ("class", "class_overlay.png"),
        ("merge", "merge_overlay.png"),
    ):
        path = out_dir / filename
        stages[stage_key].save(path)
        saved[stage_key] = str(path)

    summary = result_to_dict(result)
    summary["source_image"] = str(image_path.resolve())
    summary["saved_files"] = saved
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved["summary"] = str(summary_path)

    return saved


def _ask_image_path() -> Path:
    """Запрашивает путь к изображению, пока файл не будет найден."""
    print("Укажите путь к RGB-изображению.")
    while True:
        raw = input("Путь к фото: ").strip().strip('"').strip("'")
        if not raw:
            print("Путь не может быть пустым.")
            continue
        path = Path(raw).expanduser()
        if path.exists():
            return path
        print(f"Файл не найден: {path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Предсказание типа застройки / здания",
        epilog=(
            "Примеры:\n"
            "  python src/predict.py\n"
            "  python src/predict.py data/raw/ubc/val/tile.tif --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Путь к RGB-изображению (если не указан — будет запрос)",
    )
    parser.add_argument(
        "--mode",
        choices=("tile", "building"),
        default="tile",
        help="tile: 4-этапный пайплайн; building: один кроп здания",
    )
    parser.add_argument("--json", action="store_true", help="Вывести результат в JSON")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Папка для масок (по умолчанию reports/predictions/<имя_файла>/)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять маски и визуализации на диск",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    image_path = args.image or _ask_image_path()
    if not image_path.exists():
        print(f"Файл не найден: {image_path}", file=sys.stderr)
        return 1

    image = Image.open(image_path).convert("RGB")
    bundle = load_pipeline()

    if args.mode == "building":
        out = predict_building_crop(image, bundle)
    else:
        pipeline_result = predict_tile_pipeline(image, bundle)
        out = result_to_dict(pipeline_result)
        if not args.no_save:
            saved = save_tile_result(pipeline_result, image_path, args.save_dir)
            out["saved_files"] = saved
            print(f"\nМаска зданий: {saved['building_mask']}")
            print(f"Папка результатов: {Path(saved['building_mask']).parent}")

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for key, value in out.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

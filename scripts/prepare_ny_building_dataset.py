"""Готовит crop'ы зданий из Zenodo Building Type (small) в формате как UBC.

Источник: data/raw/NY_type_small/dataset_for_zenodo/
Назначение: data/processed_ny_building/{residential,commercial,industrial}/
  <stem>.jpg       — RGB crop
  <stem>_mask.png  — бинарная маска (255 = здание)

Маппинг классов:
  Single, Multi  -> residential
  Commercial     -> commercial
  Industrial     -> industrial

Маски в архиве отсутствуют. Приближение из метаданных CSV:
  колонка pixle_per_side / Pixels per side — сторона квадрата здания в пикселях.
  Маска = квадрат по центру crop'а (здание в Zenodo центрировано).
  Если метаданных нет — fallback: квадрат DEFAULT_MASK_FRAC * min(w, h).

После этого: python scripts/build_ny_building_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import DATA_ROOT, NY_BUILDING_CROPS_DIRNAME, NY_BUILDING_RAW_DIR  # noqa: E402

SOURCE_CLASS_MAP: dict[str, str] = {
    "Single": "residential",
    "Multi": "residential",
    "Commercial": "commercial",
    "Industrial": "industrial",
}

SKIP_SOURCE_CLASSES = {"High", "Hospital", "Schools", "metadata"}
JPEG_QUALITY = 92
DEFAULT_MASK_FRAC = 0.55  # fallback, если pixle_per_side нет в CSV


def _pixel_side_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        low = col.lower().replace(" ", "")
        if "pixel" in low and "side" in low:
            return col
    return None


def load_ny_building_metadata(raw_root: Path) -> dict[str, float]:
    """Image_Name -> pixle_per_side (float)."""
    records: dict[str, float] = {}

    for source_cls in SOURCE_CLASS_MAP:
        src_dir = raw_root / source_cls
        if not src_dir.exists():
            continue
        for csv_path in sorted(src_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            if "Image_Name" not in df.columns:
                continue
            pps_col = _pixel_side_column(df)
            if pps_col is None:
                continue
            for _, row in df.iterrows():
                name = str(row["Image_Name"]).strip()
                val = pd.to_numeric(row[pps_col], errors="coerce")
                if pd.notna(val) and val > 0:
                    records[name] = float(val)

    pred_path = raw_root / "metadata" / "output_predictions.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        if "Image_Name" in pred.columns and "pixel_per_side" in pred.columns:
            for _, row in pred.iterrows():
                name = str(row["Image_Name"]).strip()
                if name in records:
                    continue
                val = pd.to_numeric(row["pixel_per_side"], errors="coerce")
                if pd.notna(val) and val > 0:
                    records[name] = float(val)

    return records


def make_centered_square_mask(size: tuple[int, int], side_px: float) -> Image.Image:
    """Бинарная маска: белый квадрат side_px по центру изображения."""
    w, h = size
    side = int(round(min(side_px, w, h)))
    side = max(side, 8)
    left = (w - side) // 2
    top = (h - side) // 2
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((left, top, left + side, top + side), fill=255)
    return mask


def mask_side_for_image(
    image_name: str,
    size: tuple[int, int],
    metadata: dict[str, float],
    default_frac: float = DEFAULT_MASK_FRAC,
) -> tuple[float, str]:
    """Возвращает (сторона маски в px, источник: metadata|fallback)."""
    w, h = size
    if image_name in metadata:
        return metadata[image_name], "metadata"
    return min(w, h) * default_frac, "fallback"


def prepare_ny_building_crops(force: bool = False) -> dict[str, int]:
    raw_root = NY_BUILDING_RAW_DIR
    out_root = DATA_ROOT / NY_BUILDING_CROPS_DIRNAME
    metadata = load_ny_building_metadata(raw_root)

    counts: dict[str, int] = {cls: 0 for cls in ("residential", "commercial", "industrial")}
    stats = {"metadata_mask": 0, "fallback_mask": 0, "skipped_exists": 0}

    if not raw_root.exists():
        raise FileNotFoundError(f"Не найден каталог с исходниками: {raw_root}")

    for target_cls in counts:
        (out_root / target_cls).mkdir(parents=True, exist_ok=True)

    print(f"Метаданных pixle_per_side: {len(metadata)} записей")

    for source_cls, target_cls in SOURCE_CLASS_MAP.items():
        src_dir = raw_root / source_cls
        if not src_dir.exists():
            print(f"  пропуск (нет папки): {source_cls}")
            continue

        for src_path in sorted(src_dir.glob("*.tif")):
            out_img_path = out_root / target_cls / f"{src_path.stem}.jpg"
            out_mask_path = out_root / target_cls / f"{src_path.stem}_mask.png"

            if out_img_path.exists() and out_mask_path.exists() and not force:
                counts[target_cls] += 1
                stats["skipped_exists"] += 1
                continue

            with Image.open(src_path) as img:
                rgb = img.convert("RGB")
                side_px, source = mask_side_for_image(src_path.name, rgb.size, metadata)
                mask = make_centered_square_mask(rgb.size, side_px)
                rgb.save(out_img_path, format="JPEG", quality=JPEG_QUALITY)
                mask.save(out_mask_path)

            counts[target_cls] += 1
            stats["metadata_mask" if source == "metadata" else "fallback_mask"] += 1

    counts["_metadata_mask"] = stats["metadata_mask"]
    counts["_fallback_mask"] = stats["fallback_mask"]
    counts["_skipped_exists"] = stats["skipped_exists"]
    return counts


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Подготовка processed_ny_building из Zenodo small")
    parser.add_argument("--force", action="store_true", help="Перезаписать JPG и маски")
    args = parser.parse_args()

    print(f"Источник: {NY_BUILDING_RAW_DIR}")
    print(f"Назначение: {DATA_ROOT / NY_BUILDING_CROPS_DIRNAME}")
    print("Маппинг:", SOURCE_CLASS_MAP)
    print("Маска: квадрат по центру из pixle_per_side (CSV); fallback frac =", DEFAULT_MASK_FRAC)

    counts = prepare_ny_building_crops(force=args.force)
    total = sum(v for k, v in counts.items() if not k.startswith("_"))

    print(f"\nГотово: {total} пар image+mask")
    for cls in ("residential", "commercial", "industrial"):
        print(f"  {cls}: {counts[cls]}")
    print(f"  маска из metadata: {counts['_metadata_mask']}")
    print(f"  маска fallback:    {counts['_fallback_mask']}")
    if counts["_skipped_exists"]:
        print(f"  уже существовало:  {counts['_skipped_exists']} (используйте --force для перегенерации)")
    print("\nДалее: python scripts/build_ny_building_split.py")


if __name__ == "__main__":
    main()

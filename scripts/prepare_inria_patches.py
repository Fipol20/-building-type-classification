"""Нарезка INRIA train-тайлов 5000×5000 на патчи 512×512.

Источник: data/raw/inria/data/train/{images,gt}/*.tif
Назначение: data/processed_inria/patches/{images,masks}/

Патчи с долей здания < MIN_BUILDING_FRAC пропускаются (по умолчанию 1%).

После этого: python scripts/build_inria_split.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (  # noqa: E402
    DATA_ROOT,
    INRIA_MIN_BUILDING_FRAC,
    INRIA_PATCH_SIZE,
    INRIA_PATCH_STRIDE,
    INRIA_PATCHES_DIRNAME,
    INRIA_RAW_DIR,
)

JPEG_QUALITY = 92
CITY_PATTERN = re.compile(r"^([a-z-]+)(\d+)$", re.IGNORECASE)


def parse_tile_name(stem: str) -> tuple[str, str]:
    """austin1 -> ('austin', '1'); tyrol-w12 -> ('tyrol-w', '12')."""
    match = CITY_PATTERN.match(stem)
    if not match:
        raise ValueError(f"Не удалось разобрать имя тайла: {stem}")
    return match.group(1).lower(), match.group(2)


def _window_starts(size: int, patch: int, stride: int) -> list[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def prepare_inria_patches(
    force: bool = False,
    min_building_frac: float = INRIA_MIN_BUILDING_FRAC,
    patch_size: int = INRIA_PATCH_SIZE,
    stride: int = INRIA_PATCH_STRIDE,
) -> dict[str, int]:
    images_dir = INRIA_RAW_DIR / "data" / "train" / "images"
    gt_dir = INRIA_RAW_DIR / "data" / "train" / "gt"
    out_root = DATA_ROOT / INRIA_PATCHES_DIRNAME / "patches"
    out_images = out_root / "images"
    out_masks = out_root / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Не найден каталог изображений: {images_dir}")

    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    stats = {
        "tiles": 0,
        "patches_saved": 0,
        "patches_skipped_empty": 0,
        "patches_skipped_exists": 0,
    }

    image_paths = sorted(images_dir.glob("*.tif"))
    for image_path in image_paths:
        gt_path = gt_dir / image_path.name
        if not gt_path.exists():
            print(f"  пропуск (нет gt): {image_path.name}")
            continue

        city, tile_id = parse_tile_name(image_path.stem)
        stats["tiles"] += 1

        with Image.open(image_path) as img, Image.open(gt_path) as gt:
            img = img.convert("RGB")
            gt = gt.convert("L")
            w, h = img.size
            ys = _window_starts(h, patch_size, stride)
            xs = _window_starts(w, patch_size, stride)

            for y in ys:
                for x in xs:
                    patch_name = f"{city}_{tile_id}_{y}_{x}"
                    out_img_path = out_images / f"{patch_name}.jpg"
                    out_mask_path = out_masks / f"{patch_name}.png"

                    if out_img_path.exists() and out_mask_path.exists() and not force:
                        stats["patches_skipped_exists"] += 1
                        stats["patches_saved"] += 1
                        continue

                    crop_img = img.crop((x, y, x + patch_size, y + patch_size))
                    crop_gt = gt.crop((x, y, x + patch_size, y + patch_size))
                    gt_arr = np.asarray(crop_gt, dtype=np.uint8)
                    building_frac = float((gt_arr >= 128).mean())

                    if building_frac < min_building_frac:
                        stats["patches_skipped_empty"] += 1
                        continue

                    mask_bin = (gt_arr >= 128).astype(np.uint8) * 255
                    crop_img.save(out_img_path, format="JPEG", quality=JPEG_QUALITY)
                    Image.fromarray(mask_bin, mode="L").save(out_mask_path)

                    stats["patches_saved"] += 1

    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Нарезка INRIA на патчи 512×512")
    parser.add_argument("--force", action="store_true", help="Перезаписать существующие патчи")
    parser.add_argument(
        "--min-building-frac",
        type=float,
        default=INRIA_MIN_BUILDING_FRAC,
        help="Минимальная доля пикселей здания в патче",
    )
    args = parser.parse_args()

    print(f"Источник: {INRIA_RAW_DIR / 'data/train'}")
    print(f"Назначение: {DATA_ROOT / INRIA_PATCHES_DIRNAME / 'patches'}")
    print(f"patch={INRIA_PATCH_SIZE}, stride={INRIA_PATCH_STRIDE}, min_building_frac={args.min_building_frac}")

    stats = prepare_inria_patches(force=args.force, min_building_frac=args.min_building_frac)

    print(f"\nТайлов обработано: {stats['tiles']}")
    print(f"Патчей сохранено: {stats['patches_saved']}")
    print(f"Пропущено (пустые): {stats['patches_skipped_empty']}")
    if stats["patches_skipped_exists"]:
        print(f"Уже существовало: {stats['patches_skipped_exists']}")
    print("\nДалее: python scripts/build_inria_split.py")


if __name__ == "__main__":
    main()

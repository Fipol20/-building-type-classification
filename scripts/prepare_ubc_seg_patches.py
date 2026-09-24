"""Патчи 512×512 для fine-tune INRIA на UBC (только train-тайлы).

Маска: union всех COCO-полигонов зданий на тайле.
Источник: data/raw/ubc/train/*_RGB.tif + annotations/use_coarse_train.json
Назначение: data/processed_ubc_seg/patches/{images,masks}/

После: python scripts/build_ubc_seg_split.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (  # noqa: E402
    DATA_ROOT,
    INRIA_MIN_BUILDING_FRAC,
    INRIA_PATCH_SIZE,
    INRIA_PATCH_STRIDE,
    UBC_RAW_DIR,
)

UBC_SEG_DIRNAME = "processed_ubc_seg"
JPEG_QUALITY = 92
TILE_STEM_RE = re.compile(r"^(.*)_RGB$", re.IGNORECASE)


def _window_starts(size: int, patch: int, stride: int) -> list[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _tile_stem(file_name: str) -> str:
    stem = Path(file_name).stem
    m = TILE_STEM_RE.match(stem)
    return m.group(1) if m else stem


def _rasterize_tile_mask(
    width: int,
    height: int,
    annotations: list[dict],
) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for ann in annotations:
        for poly in ann.get("segmentation") or []:
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            if len(pts) >= 2:
                draw.polygon(pts, fill=255)
    return mask


def _load_train_annotations() -> tuple[dict[str, int], dict[int, list[dict]]]:
    json_path = UBC_RAW_DIR / "annotations" / "use_coarse_train.json"
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    name_to_id = {img["file_name"]: img["id"] for img in data.get("images", [])}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    return name_to_id, anns_by_image


def prepare_ubc_seg_patches(
    force: bool = False,
    patch_size: int = INRIA_PATCH_SIZE,
    stride: int = INRIA_PATCH_STRIDE,
    min_building_frac: float = INRIA_MIN_BUILDING_FRAC,
) -> dict[str, int]:
    train_dir = UBC_RAW_DIR / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Нет {train_dir}")

    out_root = DATA_ROOT / UBC_SEG_DIRNAME / "patches"
    out_images = out_root / "images"
    out_masks = out_root / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    name_to_id, anns_by_image = _load_train_annotations()
    stats = {
        "tiles": 0,
        "patches_saved": 0,
        "patches_skipped_empty": 0,
        "patches_skipped_exists": 0,
    }

    for image_path in sorted(train_dir.glob("*_RGB.tif")):
        file_name = image_path.name
        image_id = name_to_id.get(file_name)
        if image_id is None:
            continue
        tile_stem = _tile_stem(file_name)
        stats["tiles"] += 1

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            tile_mask = _rasterize_tile_mask(w, h, anns_by_image.get(image_id, []))
            ys = _window_starts(h, patch_size, stride)
            xs = _window_starts(w, patch_size, stride)

            for y in ys:
                for x in xs:
                    patch_name = f"{tile_stem}_{y}_{x}"
                    out_img = out_images / f"{patch_name}.jpg"
                    out_msk = out_masks / f"{patch_name}.png"
                    if out_img.exists() and out_msk.exists() and not force:
                        stats["patches_skipped_exists"] += 1
                        continue

                    patch_img = img.crop((x, y, x + patch_size, y + patch_size))
                    patch_mask = tile_mask.crop((x, y, x + patch_size, y + patch_size))
                    frac = np.asarray(patch_mask, dtype=np.float32).mean() / 255.0
                    if frac < min_building_frac:
                        stats["patches_skipped_empty"] += 1
                        continue

                    patch_img.save(out_img, quality=JPEG_QUALITY)
                    patch_mask.save(out_msk)
                    stats["patches_saved"] += 1

    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="UBC train-тайлы → патчи для INRIA fine-tune")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print("=== UBC seg patches (train only) ===")
    stats = prepare_ubc_seg_patches(force=args.force)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("\nДалее: python scripts/build_ubc_seg_split.py")


if __name__ == "__main__":
    main()

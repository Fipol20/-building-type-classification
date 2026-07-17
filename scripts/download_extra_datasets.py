"""Скачивает дополнительные датасеты и раскладывает по папкам.

- INRIA Aerial Image Labeling (building / not building) -> data/raw/inria/
- UBC building crops + маски (residential / commercial / industrial) -> data/processed_ubc/

Для каждого здания сохраняется пара файлов:
  <cls>/<stem>.jpg       - кроп bbox+padding (контекст вокруг здания)
  <cls>/<stem>_mask.png  - бинарная маска (255 = пиксели этого здания, растеризована из COCO segmentation)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INRIA_DIR = DATA_DIR / "raw" / "inria"
UBC_RAW_DIR = DATA_DIR / "raw" / "ubc"
UBC_CROPS_DIR = DATA_DIR / "processed_ubc"

UBC_TARGET_CLASSES = {"residential", "commercial", "industrial"}
UBC_CATEGORY_MAP = {
    1: "residential",
    2: "commercial",
    3: "industrial",
}
CROP_PADDING = 16
MIN_CROP_SIZE = 32


def download_inria(full: bool = False) -> None:
    from huggingface_hub import snapshot_download

    patterns = ["data/train/**"]
    if full:
        patterns.append("data/test/**")

    print("=== INRIA Aerial Image Labeling (HuggingFace) ===")
    print(f"  Назначение: {INRIA_DIR}")
    print(f"  Режим: {'train + test' if full else 'train only'}")

    snapshot_download(
        repo_id="blanchon/INRIA-Aerial-Image-Labeling",
        repo_type="dataset",
        local_dir=INRIA_DIR,
        allow_patterns=patterns,
    )

    train_images = list((INRIA_DIR / "data" / "train" / "images").glob("*.tif"))
    train_gt = list((INRIA_DIR / "data" / "train" / "gt").glob("*.tif"))
    print(f"  train/images: {len(train_images)}")
    print(f"  train/gt:     {len(train_gt)}")

    if full:
        test_images = list((INRIA_DIR / "data" / "test" / "images").glob("*.tif"))
        print(f"  test/images:  {len(test_images)}")


def _find_ubc_image(stem: str, split: str) -> Path | None:
    for suffix in ("_RGB.tif", ".tif"):
        path = UBC_RAW_DIR / split / f"{stem}{suffix}" if suffix.startswith("_") else UBC_RAW_DIR / split / f"{stem}{suffix}"
        if path.exists():
            return path
    # file_name already includes _RGB.tif
    direct = UBC_RAW_DIR / split / stem
    if direct.exists():
        return direct
    return None


def _crop_box(img_size: tuple[int, int], bbox: list[float], padding: int) -> tuple[int, int, int, int] | None:
    """Прямоугольник bbox+padding, обрезанный по границам изображения."""
    img_w, img_h = img_size
    x, y, w, h = bbox
    left = max(0, int(x) - padding)
    top = max(0, int(y) - padding)
    right = min(img_w, int(x + w) + padding)
    bottom = min(img_h, int(y + h) + padding)
    if right - left < MIN_CROP_SIZE or bottom - top < MIN_CROP_SIZE:
        return None
    return left, top, right, bottom


def _rasterize_mask(segmentation: list[list[float]], box: tuple[int, int, int, int]) -> Image.Image:
    """Растеризует COCO-полигон(ы) здания в бинарную маску размера кропа (255 = здание)."""
    left, top, right, bottom = box
    mask = Image.new("L", (right - left, bottom - top), 0)
    draw = ImageDraw.Draw(mask)
    for poly in segmentation:
        pts = [(poly[i] - left, poly[i + 1] - top) for i in range(0, len(poly), 2)]
        if len(pts) >= 2:
            draw.polygon(pts, fill=255)
    return mask


def extract_ubc_crops(force: bool = False) -> dict[str, int]:
    counts = {cls: 0 for cls in UBC_TARGET_CLASSES}
    image_cache: dict[tuple[str, str], Image.Image] = {}

    for cls in UBC_TARGET_CLASSES:
        (UBC_CROPS_DIR / cls).mkdir(parents=True, exist_ok=True)

    ann_dir = UBC_RAW_DIR / "annotations"
    if not ann_dir.exists():
        raise FileNotFoundError(f"UBC annotations not found: {ann_dir}")

    for json_path in sorted(ann_dir.glob("use_coarse_*.json")):
        split = "train" if "_train" in json_path.stem else "val"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        images_by_id = {img["id"]: img for img in data.get("images", [])}

        for ann in data.get("annotations", []):
            cls = UBC_CATEGORY_MAP.get(ann["category_id"])
            if cls is None:
                continue

            img_meta = images_by_id.get(ann["image_id"])
            if img_meta is None:
                continue

            stem = f"{split}_{ann['image_id']:04d}_{ann['id']:05d}"
            out_img_path = UBC_CROPS_DIR / cls / f"{stem}.jpg"
            out_mask_path = UBC_CROPS_DIR / cls / f"{stem}_mask.png"
            if out_img_path.exists() and out_mask_path.exists() and not force:
                counts[cls] += 1
                continue

            file_name = img_meta["file_name"]
            cache_key = (split, file_name)
            if cache_key not in image_cache:
                img_path = UBC_RAW_DIR / split / file_name
                if not img_path.exists():
                    img_stem = Path(file_name).stem.replace("_RGB", "")
                    img_path = _find_ubc_image(img_stem, split) or _find_ubc_image(file_name, split)
                if img_path is None or not img_path.exists():
                    continue
                image_cache[cache_key] = Image.open(img_path).convert("RGB")

            source_img = image_cache[cache_key]
            box = _crop_box(source_img.size, ann["bbox"], CROP_PADDING)
            if box is None:
                continue

            segmentation = ann.get("segmentation") or []
            if not segmentation:
                continue

            crop = source_img.crop(box)
            mask = _rasterize_mask(segmentation, box)

            crop.save(out_img_path, quality=95)
            mask.save(out_mask_path)
            counts[cls] += 1

    return counts


def print_layout() -> None:
    print("\n=== Структура data/ ===")
    dirs = [
        DATA_DIR / "processed",
        DATA_DIR / "processed_ubc",
        DATA_DIR / "raw" / "inria",
        DATA_DIR / "raw" / "ubc",
    ]
    for d in dirs:
        if not d.exists():
            print(f"  {d.relative_to(PROJECT_ROOT)}: (нет)")
            continue
        if d.name in {"processed", "processed_ubc"}:
            sub = [p.name for p in d.iterdir() if p.is_dir()]
            print(f"  {d.relative_to(PROJECT_ROOT)}/: {', '.join(sorted(sub))}")
        else:
            n_files = sum(1 for _ in d.rglob("*") if _.is_file())
            print(f"  {d.relative_to(PROJECT_ROOT)}/: {n_files} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download INRIA and extract UBC crops.")
    parser.add_argument("--inria", action="store_true", help="Download INRIA from HuggingFace")
    parser.add_argument("--inria-full", action="store_true", help="Download INRIA train + test")
    parser.add_argument("--ubc-crops", action="store_true", help="Extract UBC building crops")
    parser.add_argument("--force-ubc", action="store_true", help="Re-extract UBC crops")
    parser.add_argument("--all", action="store_true", help="INRIA train + UBC crops")
    args = parser.parse_args()

    if not any([args.inria, args.inria_full, args.ubc_crops, args.all]):
        args.all = True

    if args.inria or args.inria_full or args.all:
        download_inria(full=args.inria_full)

    if args.ubc_crops or args.all:
        print("\n=== UBC crops -> data/processed_ubc/ ===")
        counts = extract_ubc_crops(force=args.force_ubc)
        for cls, n in sorted(counts.items()):
            print(f"  {cls}: {n}")

    print_layout()


if __name__ == "__main__":
    main()

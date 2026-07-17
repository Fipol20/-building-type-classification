"""Скачивает 3 класса (residential, commercial, industrial) в data/raw/.

UBC v1 с Baidu Pan автоматически не скачивается (зашифрованная ссылка).
Пока используем близкие классы из NWPU-RESISC45 через HuggingFace.

Если вручную положить UBC_v1.0.zip в data/raw/ubc/, скрипт также
распакует архив и сохранит COCO-аннотации use_coarse для этих классов.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
UBC_DIR = RAW_DIR / "ubc"
UBC_ARCHIVE = UBC_DIR / "UBC_v1.0.zip"
UBC_EXTRACT = UBC_DIR / "extracted"

TARGET_CLASSES = ("residential", "commercial", "industrial")

RESISC45_MAP = {
    "commercial_area": "commercial",
    "industrial_area": "industrial",
    "dense_residential": "residential",
    "sparse_residential": "residential",
    "medium_residential": "residential",
}


def download_resisc45_classes() -> dict[str, int]:
    from datasets import load_dataset

    counts = {cls: 0 for cls in TARGET_CLASSES}
    for cls in TARGET_CLASSES:
        (RAW_DIR / cls).mkdir(parents=True, exist_ok=True)

    print("Загрузка RESISC45 (timm/resisc45)...")
    ds = load_dataset("timm/resisc45", split="train")
    label_names = ds.features["label"].names

    for i, row in enumerate(ds):
        src_label = label_names[row["label"]]
        target = RESISC45_MAP.get(src_label)
        if target is None:
            continue

        dst = RAW_DIR / target / f"resisc45_{i:05d}.jpg"
        if not dst.exists():
            row["image"].convert("RGB").save(dst, quality=95)
        counts[target] += 1

    return counts


def process_ubc_archive() -> dict[str, int] | None:
    if not UBC_ARCHIVE.exists():
        return None

    if not UBC_EXTRACT.exists() or not any(UBC_EXTRACT.iterdir()):
        print(f"Распаковка {UBC_ARCHIVE.name}...")
        UBC_EXTRACT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(UBC_ARCHIVE, "r") as zf:
            zf.extractall(UBC_EXTRACT)

    counts = {cls: 0 for cls in TARGET_CLASSES}
    coco_dir = UBC_DIR / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    for json_path in sorted(UBC_EXTRACT.rglob("*.json")):
        name = json_path.name.lower()
        if "use" not in name and "function" not in name:
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        cat_id_to_name = {c["id"]: c["name"].lower() for c in data.get("categories", [])}
        img_by_id = {img["id"]: img for img in data.get("images", [])}

        filtered = {
            "info": data.get("info", {}),
            "licenses": data.get("licenses", []),
            "categories": [c for c in data.get("categories", []) if c["name"].lower() in TARGET_CLASSES],
            "images": [],
            "annotations": [],
        }
        kept_image_ids: set[int] = set()

        for ann in data.get("annotations", []):
            cls = cat_id_to_name.get(ann["category_id"], "")
            if cls not in TARGET_CLASSES:
                continue
            filtered["annotations"].append(ann)
            kept_image_ids.add(ann["image_id"])
            counts[cls] += 1

        for img_id in kept_image_ids:
            if img_id in img_by_id:
                filtered["images"].append(img_by_id[img_id])

        out_json = coco_dir / json_path.name
        out_json.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")

        for cls in TARGET_CLASSES:
            (RAW_DIR / cls / "ubc").mkdir(parents=True, exist_ok=True)

    return counts


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=== RESISC45 -> data/raw/ ===")
    resisc_counts = download_resisc45_classes()
    for cls, n in sorted(resisc_counts.items()):
        print(f"  {cls}: {n}")

    ubc_counts = process_ubc_archive()
    if ubc_counts:
        print("\n=== UBC COCO-аннотации (use_coarse) ===")
        for cls, n in sorted(ubc_counts.items()):
            print(f"  {cls}: {n} building instances")
        print(f"  JSON: {UBC_DIR / 'coco'}")
    else:
        print("\nUBC архив не найден.")
        print(f"  Положите UBC_v1.0.zip сюда: {UBC_ARCHIVE}")
        print("  Baidu: https://pan.baidu.com/s/1M6yYD1lvbqsVpn5MHGa2tg?pwd=7hbm")

    print("\nГотово. Классы лежат в data/raw/{residential,commercial,industrial}/")


if __name__ == "__main__":
    main()

"""Строит data/processed_inria/split.csv — split по городам (не по патчам).

train: austin, chicago, kitsap
val:   vienna
test:  tyrol-w

Предпосылка: python scripts/prepare_inria_patches.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import save_split  # noqa: E402
from utils import (  # noqa: E402
    DATA_ROOT,
    INRIA_CITY_SPLIT,
    INRIA_PATCHES_DIRNAME,
    INRIA_SPLIT_CSV,
)

PATCH_NAME_PATTERN = re.compile(r"^([a-z-]+)_(\d+)_(\d+)_(\d+)$", re.IGNORECASE)


def collect_inria_patches(patches_dir: Path) -> pd.DataFrame:
    images_dir = patches_dir / "images"
    masks_dir = patches_dir / "masks"
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Не найдено {images_dir}. Сначала: python scripts/prepare_inria_patches.py"
        )

    city_to_split = {}
    for split, cities in INRIA_CITY_SPLIT.items():
        for city in cities:
            city_to_split[city.lower()] = split

    records = []
    for image_path in sorted(images_dir.glob("*.jpg")):
        match = PATCH_NAME_PATTERN.match(image_path.stem)
        if not match:
            continue
        city = match.group(1).lower()
        mask_path = masks_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            continue
        split = city_to_split.get(city)
        if split is None:
            print(f"  пропуск (неизвестный город): {image_path.name}")
            continue
        records.append(
            {
                "path": str(image_path),
                "mask_path": str(mask_path),
                "city": city,
                "split": split,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    patches_dir = DATA_ROOT / INRIA_PATCHES_DIRNAME / "patches"
    df = collect_inria_patches(patches_dir)
    if len(df) == 0:
        raise FileNotFoundError(f"Не найдено патчей в {patches_dir}")

    print(f"Всего патчей: {len(df)}")
    print("\nПо split:")
    print(df.groupby("split").size().to_string())
    print("\nПо split и городу:")
    print(df.groupby(["split", "city"]).size().to_string())

    save_split(df, INRIA_SPLIT_CSV)
    print(f"\nСохранено: {INRIA_SPLIT_CSV}")


if __name__ == "__main__":
    main()

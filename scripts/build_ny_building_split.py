"""Строит data/processed_ny_building/split.csv (70/15/15, стратификация по классу).

Предпосылка: python scripts/prepare_ny_building_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import make_split, save_split  # noqa: E402
from utils import (  # noqa: E402
    DATA_ROOT,
    NY_BUILDING_CLASSES,
    NY_BUILDING_CROPS_DIRNAME,
    NY_BUILDING_SPLIT_CSV,
    RANDOM_SEED,
)


def collect_ny_crops(crops_dir: Path, classes: list[str] = NY_BUILDING_CLASSES) -> pd.DataFrame:
    records = []
    for cls in classes:
        cls_dir = crops_dir / cls
        if not cls_dir.exists():
            continue
        for path in sorted(cls_dir.glob("*.jpg")):
            records.append(
                {
                    "path": str(path),
                    "class": cls,
                    "source": "NY_Zenodo",
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    crops_dir = DATA_ROOT / NY_BUILDING_CROPS_DIRNAME
    df = collect_ny_crops(crops_dir)
    if len(df) == 0:
        raise FileNotFoundError(
            f"Не найдено *.jpg в {crops_dir}. "
            "Сначала: python scripts/prepare_ny_building_dataset.py"
        )

    print(f"Всего crop'ов: {len(df)}")
    print("\nПо классам:")
    print(df["class"].value_counts().to_string())

    split_df = make_split(df, seed=RANDOM_SEED)
    save_split(split_df, NY_BUILDING_SPLIT_CSV)

    print("\nРазбиение по split:")
    print(split_df.groupby("split").size().to_string())
    print("\nПо split и классу:")
    print(split_df.groupby(["split", "class"]).size().to_string())
    print(f"\nСохранено: {NY_BUILDING_SPLIT_CSV}")


if __name__ == "__main__":
    main()

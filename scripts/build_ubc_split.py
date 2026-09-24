"""Строит data/processed_ubc/split.csv для building-классификатора (3 класса UBC).

В отличие от build_split.py (AID+RESISC45), здесь исходное разбиение UBC
train/val (закодированное в префиксе имени файла) используется как основа:
  - исходный UBC val -> итоговый test (снимки, которых модель не видела)
  - исходный UBC train -> делится на train/val (85/15), стратифицированно по классу

Дисбаланс классов — свойство разметки use_coarse (не ошибка сплита), примерно:
  residential ~33.5k | commercial ~4.9k | industrial ~0.76k (~1.9%).
Классы public/other из UBC намеренно не включаются.

Каждому зданию соответствует пара файлов <stem>.jpg (кроп) и <stem>_mask.png
(бинарная маска); split.csv ссылается только на *.jpg.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import save_split  # noqa: E402
from utils import DATA_ROOT, RANDOM_SEED, UBC_CLASSES, UBC_CROPS_DIRNAME, UBC_SPLIT_CSV  # noqa: E402

TRAIN_VAL_SPLIT = 0.85  # доля исходного UBC train, идущая в итоговый train


def collect_ubc_crops(crops_dir: Path, classes: list[str] = UBC_CLASSES) -> pd.DataFrame:
    """Собирает path/class/orig_split по *.jpg (маски *_mask.png не включаются)."""
    records = []
    for cls in classes:
        cls_dir = crops_dir / cls
        if not cls_dir.exists():
            continue
        for path in sorted(cls_dir.glob("*.jpg")):
            orig_split = path.stem.split("_")[0]  # "train" или "val" (исходный UBC split)
            records.append(
                {
                    "path": str(path),
                    "class": cls,
                    "source": "UBC",
                    "orig_split": orig_split,
                }
            )
    return pd.DataFrame(records)


def make_ubc_split(df: pd.DataFrame, train_val_split: float = TRAIN_VAL_SPLIT, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """test = исходный UBC val; train/val = стратифицированное разбиение исходного UBC train."""
    test_df = df[df["orig_split"] == "val"].copy()
    train_orig_df = df[df["orig_split"] == "train"].copy()

    train_df, val_df = train_test_split(
        train_orig_df,
        train_size=train_val_split,
        stratify=train_orig_df["class"],
        random_state=seed,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return out.drop(columns=["orig_split"])


def main() -> None:
    crops_dir = DATA_ROOT / UBC_CROPS_DIRNAME
    df = collect_ubc_crops(crops_dir)
    if len(df) == 0:
        raise FileNotFoundError(
            f"Не найдено *.jpg в {crops_dir}. "
            "Запустите: python scripts/download_extra_datasets.py --ubc-crops"
        )

    print(f"Всего crop'ов зданий: {len(df)}")
    print("\nПо классам (весь набор):")
    print(df["class"].value_counts().to_string())

    split_df = make_ubc_split(df)
    save_split(split_df, UBC_SPLIT_CSV)

    print("\nРазбиение по split:")
    print(split_df.groupby("split").size().to_string())
    print("\nПо split и классу:")
    print(split_df.groupby(["split", "class"]).size().to_string())
    print(f"\nСохранено: {UBC_SPLIT_CSV}")


if __name__ == "__main__":
    main()

"""split.csv для processed_ubc_seg: 85% train / 15% val по тайлам (без test leakage)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import save_split  # noqa: E402
from utils import DATA_ROOT, RANDOM_SEED  # noqa: E402

UBC_SEG_DIRNAME = "processed_ubc_seg"
PATCH_RE = re.compile(r"^(.+)_(\d+)_(\d+)$")


def collect_patches(patches_dir: Path) -> pd.DataFrame:
    images_dir = patches_dir / "images"
    masks_dir = patches_dir / "masks"
    records = []
    for image_path in sorted(images_dir.glob("*.jpg")):
        m = PATCH_RE.match(image_path.stem)
        if not m:
            continue
        tile_stem = m.group(1)
        mask_path = masks_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            continue
        records.append(
            {
                "path": str(image_path),
                "mask_path": str(mask_path),
                "tile": tile_stem,
                "source": "UBC",
            }
        )
    if not records:
        raise FileNotFoundError(f"Нет патчей в {images_dir}. Запустите prepare_ubc_seg_patches.py")
    return pd.DataFrame(records)


def make_split(df: pd.DataFrame, val_frac: float = 0.15, seed: int = RANDOM_SEED) -> pd.DataFrame:
    tiles = sorted(df["tile"].unique())
    train_tiles, val_tiles = train_test_split(tiles, test_size=val_frac, random_state=seed)
    train_set, val_set = set(train_tiles), set(val_tiles)
    out = df.copy()
    out["split"] = out["tile"].apply(lambda t: "val" if t in val_set else "train")
    return out


def main() -> None:
    patches_dir = DATA_ROOT / UBC_SEG_DIRNAME / "patches"
    out_csv = DATA_ROOT / UBC_SEG_DIRNAME / "split.csv"
    df = collect_patches(patches_dir)
    split_df = make_split(df)
    save_split(split_df, out_csv)
    print(f"Патчей: {len(split_df)}")
    print(split_df.groupby("split").size().to_string())
    print(f"Сохранено: {out_csv}")


if __name__ == "__main__":
    main()

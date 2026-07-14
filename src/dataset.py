"""Разбиение данных и загрузчик для PyTorch."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from utils import (
    AUG_COLOR_JITTER,
    AUG_RANDOM_RESIZED_CROP_SCALE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    RANDOM_SEED,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
    project_root,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASSES = ["commercial", "industrial", "dense_residential", "sparse_residential"]

PROCESSED_DIRNAME = "processed"

SOURCE_AID = "AID"
SOURCE_RESISC45 = "NWPU-RESISC45"


def infer_source(filename: str) -> str:
    """Определяет источник изображения (AID / RESISC45) по имени файла."""
    return SOURCE_RESISC45 if filename.lower().startswith("resisc45") else SOURCE_AID


def build_dataframe(
    data_dir: Path,
    classes: list[str] = CLASSES,
    source: str | None = None,
) -> pd.DataFrame:
    """Собирает таблицу path/class[/source] по изображениям в каталоге."""
    records = []
    for cls in classes:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue
        for path in sorted(cls_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                rec = {
                    "path": str(path),
                    "class": cls,
                    "source": source or infer_source(path.name),
                }
                records.append(rec)
    return pd.DataFrame(records)


def build_combined_dataframe(
    data_root: Path,
    classes: list[str] = CLASSES,
    processed_dirname: str = PROCESSED_DIRNAME,
) -> pd.DataFrame:
    processed_dir = data_root / processed_dirname
    df = build_dataframe(processed_dir, classes=classes)
    if len(df) == 0:
        raise FileNotFoundError(f"Не найдено изображений в {processed_dir}")
    return df


def make_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Разбиение 70/15/15 по классам."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    stratify_col = df["class"]
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_frac + test_frac),
        stratify=stratify_col,
        random_state=seed,
    )
    val_ratio_within_temp = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_within_temp),
        stratify=temp_df["class"],
        random_state=seed,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def get_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Изменение размера и нормализация под ImageNet."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_train_transforms(image_size: int = IMAGE_SIZE, augment: bool = True) -> transforms.Compose:
    """Преобразования для обучающей выборки."""
    if not augment:
        return get_transforms(image_size)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=AUG_RANDOM_RESIZED_CROP_SCALE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(**AUG_COLOR_JITTER),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class BuildingDataset(Dataset):
    """Набор данных по таблице path/class/split."""

    def __init__(self, df: pd.DataFrame, split: str, classes: list[str] = CLASSES, transform=None):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform or get_transforms()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[row["class"]]
        return image, label


def save_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = project_root()
    out = df.copy()
    out["path"] = out["path"].apply(lambda p: _to_repo_relative(p, root))
    out.to_csv(path, index=False)


def load_split(path: Path) -> pd.DataFrame:
    """Загружает таблицу с разбиением из CSV."""
    df = pd.read_csv(path)
    root = project_root()
    df["path"] = df["path"].apply(lambda p: _to_absolute(p, root))
    return df


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Сводка: количество изображений по source и class."""
    if "source" not in df.columns:
        return df.groupby("class").size().reset_index(name="count")
    return (
        df.groupby(["source", "class"])
        .size()
        .reset_index(name="count")
        .sort_values(["source", "class"])
    )

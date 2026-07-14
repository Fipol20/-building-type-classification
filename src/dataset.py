"""Разбиение данных и загрузчик для PyTorch."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

RANDOM_SEED = 42
IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Среднее и СКО ImageNet (для предобученных моделей)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASSES = ["commercial", "industrial", "dense_residential", "sparse_residential"]

# Источники данных: AID + RESISC45
DATASET_SOURCES = {
    "aid": "AID",
    "resisc45": "NWPU-RESISC45",
}


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
                rec = {"path": str(path), "class": cls}
                if source:
                    rec["source"] = source
                records.append(rec)
    return pd.DataFrame(records)


def build_combined_dataframe(
    data_root: Path,
    sources: dict[str, str] | None = None,
    classes: list[str] = CLASSES,
) -> pd.DataFrame:
    """Собирает DataFrame из нескольких датасетов (aid, resisc45)."""
    sources = sources or DATASET_SOURCES
    frames = []
    for folder, source_name in sources.items():
        data_dir = data_root / folder
        if not data_dir.exists():
            continue
        df = build_dataframe(data_dir, classes=classes, source=source_name)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Не найдено изображений в {data_root}")
    return pd.concat(frames, ignore_index=True)


def make_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Стратифицированное разбиение 70/15/15 по классам."""
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
    """Изменение размера, тензор и нормализация под ImageNet."""
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
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    """Сохраняет таблицу с разбиением в CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_split(path: Path) -> pd.DataFrame:
    """Загружает таблицу с разбиением из CSV."""
    return pd.read_csv(path)


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

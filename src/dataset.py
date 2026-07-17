"""Разбиение данных и загрузчик для PyTorch."""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from utils import (
    AUG_COLOR_JITTER,
    AUG_RANDOM_RESIZED_CROP_SCALE,
    BUILDING_AUG_AFFINE_DEGREES,
    BUILDING_AUG_AFFINE_TRANSLATE,
    BUILDING_AUG_COLOR_JITTER,
    BUILDING_AUG_ROTATION_DEG,
    BUILDING_MASK_BG_GRAY,
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

# INRIA segmentation patches (дублируется в utils.INRIA_PATCH_SIZE)
DEFAULT_INRIA_PATCH_SIZE = 512

SOURCE_AID = "AID"
SOURCE_RESISC45 = "NWPU-RESISC45"


def _to_repo_relative(path: str, root: Path) -> str:
    """Путь относительно корня репозитория."""
    p = Path(path)
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def _to_absolute(path: str, root: Path) -> str:
    """Абсолютный путь."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(root / p)


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
    """Преобразования для обучающей выборки (zone-классификатор)."""
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


def get_building_train_transforms(image_size: int = IMAGE_SIZE, augment: bool = True) -> transforms.Compose:
    """Аугментации для building-классификатора (сильнее, чем zone)."""
    if not augment:
        return get_transforms(image_size)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=AUG_RANDOM_RESIZED_CROP_SCALE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(BUILDING_AUG_ROTATION_DEG),
            transforms.RandomAffine(
                degrees=BUILDING_AUG_AFFINE_DEGREES,
                translate=BUILDING_AUG_AFFINE_TRANSLATE,
            ),
            transforms.ColorJitter(**BUILDING_AUG_COLOR_JITTER),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def apply_building_mask(
    image: Image.Image,
    mask: Image.Image | None,
    bg_gray: int = BUILDING_MASK_BG_GRAY,
) -> Image.Image:
    """Затирает фон вне маски здания серым — модель видит форму здания."""
    if mask is None:
        return image
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    m = np.asarray(mask.convert("L").resize(image.size, Image.NEAREST), dtype=np.uint8)
    outside = m < 128
    rgb[outside] = bg_gray
    return Image.fromarray(rgb, mode="RGB")


def mask_path_for_crop(crop_path: str | Path) -> Path:
    """Путь к *_mask.png рядом с crop *.jpg."""
    p = Path(crop_path)
    return p.with_name(f"{p.stem}_mask.png")


class BuildingDataset(Dataset):
    """Crop'ы зданий UBC: path/class/split + опциональная маска (серый фон)."""

    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        classes: list[str] = CLASSES,
        transform=None,
        use_mask: bool = True,
    ):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform or get_transforms()
        self.use_mask = use_mask

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.use_mask:
            mpath = mask_path_for_crop(row["path"])
            mask = Image.open(mpath) if mpath.exists() else None
            image = apply_building_mask(image, mask)
        image = self.transform(image)
        label = self.class_to_idx[row["class"]]
        return image, label


def save_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = project_root()
    out = df.copy()
    out["path"] = out["path"].apply(lambda p: _to_repo_relative(p, root))
    if "mask_path" in out.columns:
        out["mask_path"] = out["mask_path"].apply(lambda p: _to_repo_relative(p, root))
    out.to_csv(path, index=False)


def load_split(path: Path) -> pd.DataFrame:
    """Загружает таблицу с разбиением из CSV."""
    df = pd.read_csv(path)
    root = project_root()
    df["path"] = df["path"].apply(lambda p: _to_absolute(p, root))
    return df


def load_inria_split(path: Path) -> pd.DataFrame:
    """Загружает INRIA split.csv с абсолютными path и mask_path."""
    df = pd.read_csv(path)
    root = project_root()
    df["path"] = df["path"].apply(lambda p: _to_absolute(p, root))
    df["mask_path"] = df["mask_path"].apply(lambda p: _to_absolute(p, root))
    return df


def _normalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image - mean) / std


def _apply_paired_geom_aug(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Одинаковые flip/rotate90 для image и mask."""
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    k = random.randint(0, 3)
    if k:
        angle = 90 * k
        image = image.rotate(angle, expand=False)
        mask = mask.rotate(angle, expand=False)
    return image, mask


class InriaSegmentationDataset(Dataset):
    """Патчи INRIA: image + бинарная mask (float 0/1), shape mask [1, H, W]."""

    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        augment: bool = False,
        patch_size: int = DEFAULT_INRIA_PATCH_SIZE,
    ):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.augment = augment
        self.patch_size = patch_size
        self.color_jitter = transforms.ColorJitter(**AUG_COLOR_JITTER) if augment else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")

        if image.size != (self.patch_size, self.patch_size):
            image = image.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            mask = mask.resize((self.patch_size, self.patch_size), Image.NEAREST)

        if self.augment:
            image, mask = _apply_paired_geom_aug(image, mask)
            if self.color_jitter is not None:
                image = self.color_jitter(image)

        image_t = transforms.ToTensor()(image)
        image_t = _normalize_image_tensor(image_t)
        mask_t = torch.from_numpy(
            (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.float32)
        ).unsqueeze(0)
        return image_t, mask_t


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

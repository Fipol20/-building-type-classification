"""Единая конфигурация проекта."""


from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# Пути
PATH_DATA = "data"
PATH_MODELS = "models"
PATH_REPORTS = "reports"
PATH_SPLIT_CSV = "data/processed/split.csv"
PATH_UBC_SPLIT_CSV = "data/processed_ubc/split.csv"
PATH_UBC_RAW_DIR = "data/raw/ubc"
PATH_NY_BUILDING_SPLIT_CSV = "data/processed_ny_building/split.csv"
PATH_NY_BUILDING_RAW_DIR = "data/raw/NY_type_small/dataset_for_zenodo"
PATH_INRIA_SPLIT_CSV = "data/processed_inria/split.csv"
PATH_UBC_SEG_SPLIT_CSV = "data/processed_ubc_seg/split.csv"
PATH_INRIA_RAW_DIR = "data/raw/inria"


def project_root() -> Path:
    """Корень репозитория"""
    return Path(__file__).resolve().parent.parent


def repo_path(relative: str) -> Path:
    """Собирает путь от корня репозитория."""
    return project_root() / relative


DATA_ROOT = repo_path(PATH_DATA)
MODELS_DIR = repo_path(PATH_MODELS)
REPORTS_DIR = repo_path(PATH_REPORTS)
SPLIT_CSV = repo_path(PATH_SPLIT_CSV)
UBC_SPLIT_CSV = repo_path(PATH_UBC_SPLIT_CSV)
UBC_RAW_DIR = repo_path(PATH_UBC_RAW_DIR)
NY_BUILDING_SPLIT_CSV = repo_path(PATH_NY_BUILDING_SPLIT_CSV)
NY_BUILDING_RAW_DIR = repo_path(PATH_NY_BUILDING_RAW_DIR)
INRIA_SPLIT_CSV = repo_path(PATH_INRIA_SPLIT_CSV)
UBC_SEG_SPLIT_CSV = repo_path(PATH_UBC_SEG_SPLIT_CSV)
INRIA_RAW_DIR = repo_path(PATH_INRIA_RAW_DIR)

# Общие
RANDOM_SEED = 42
IMAGE_SIZE = 224

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

NUM_WORKERS = 0
PLOT_DPI = 100
FIGURE_SIZE = (10, 6)

# Нормализация ImageNet (для предобученных моделей)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Аугментации обучающей выборки
AUG_RANDOM_RESIZED_CROP_SCALE = (0.8, 1.0)
AUG_COLOR_JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.2)

# Building-классификатор: сильнее аугментации + серый фон вне маски здания
BUILDING_AUG_COLOR_JITTER = dict(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
BUILDING_AUG_ROTATION_DEG = 15
BUILDING_AUG_AFFINE_DEGREES = 10
BUILDING_AUG_AFFINE_TRANSLATE = (0.05, 0.05)
BUILDING_MASK_BG_GRAY = 128

# Гибридный пайплайн: карта застройки (zone) + маски зданий (building), см. практика_ml.md
UBC_CROPS_DIRNAME = "processed_ubc"
UBC_CLASSES = ["residential", "commercial", "industrial"]

# Zenodo Building Type (small): Single/Multi -> residential
NY_BUILDING_CROPS_DIRNAME = "processed_ny_building"
NY_BUILDING_CLASSES = ["residential", "commercial", "industrial"]

# INRIA Aerial Image Labeling: бинарная сегментация зданий
INRIA_PATCHES_DIRNAME = "processed_inria"
UBC_SEG_DIRNAME = "processed_ubc_seg"
INRIA_PATCH_SIZE = 512
INRIA_PATCH_STRIDE = 256
INRIA_MIN_BUILDING_FRAC = 0.01
INRIA_CITY_SPLIT: dict[str, list[str]] = {
    "train": ["austin", "chicago", "kitsap"],
    "val": ["vienna"],
    "test": ["tyrol-w"],
}

# Классы карты застройки (zone) — совпадают с dataset.CLASSES, дублируются здесь
# как единая точка правды для палитр и модулей zone_map/building_masks/merge_maps.
ZONE_CLASSES = ["commercial", "industrial", "dense_residential", "sparse_residential"]

# Палитры RGB (0-255) для overlay-визуализаций
ZONE_COLORS: dict[str, tuple[int, int, int]] = {
    "commercial": (230, 25, 75),
    "industrial": (245, 130, 48),
    "dense_residential": (60, 100, 220),
    "sparse_residential": (110, 190, 250),
}
BUILDING_COLORS: dict[str, tuple[int, int, int]] = {
    "residential": (110, 190, 250),
    "commercial": (230, 25, 75),
    "industrial": (245, 130, 48),
}


@dataclass(frozen=True)
class EDAConfig:
    figure_size: tuple[int, int] = FIGURE_SIZE
    font_size: int = 11
    samples_per_class: int = 4
    subplot_width: int = 14
    subplot_height_per_class: int = 3


@dataclass(frozen=True)
class BaselineConfig:
    batch_size: int = 32
    max_epochs: int = 100
    loss_eps: float = 1e-4
    patience: int = 7
    min_epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    model_filename: str = "baseline_model.pth"
    training_curves_plot: str = "baseline_training_curves.png"
    confusion_matrix_plot: str = "baseline_confusion_matrix.png"


@dataclass(frozen=True)
class ConvNeXtConfig:
    batch_size: int = 32
    max_epochs: int = 100
    loss_eps: float = 1e-4
    patience: int = 7
    min_epochs: int = 25

    lr_candidates: list[float] = field(default_factory=lambda: [1e-3, 1e-4, 1e-5])
    stage1_max_epochs: int = 15
    stage1_min_epochs: int = 5
    stage1_patience: int = 5
    # этап 2: (сколько блоков разморозить, во сколько раз уменьшить LR)
    stage2_steps: list[tuple[int, int]] = field(
        default_factory=lambda: [(2, 10), (4, 20), (4, 50)]
    )
    stage2_min_epochs_floor: int = 5
    weight_decay: float = 1e-4

    n_correct_per_class: int = 2
    figure_size: tuple[int, int] = FIGURE_SIZE

    model_filename: str = "convnext_best.pth"
    training_curves_plot: str = "convnext_training_curves.png"
    confusion_matrix_plot: str = "convnext_confusion_matrix.png"
    correct_examples_plot: str = "convnext_correct_examples.png"
    error_examples_plot: str = "convnext_error_examples.png"


@dataclass(frozen=True)
class UBCBuildingConfig:
    """Обучение building-классификатора (3 класса) на crop'ах UBC.

    Дисбаланс UBC (примерно): residential ~33.5k, commercial ~4.9k, industrial ~0.76k
    (~1.9%). Industrial мало в самой разметке use_coarse — не потеря при сплите.
    public/other (~14k) намеренно не используются.
    UBC crop'ов в 11+ раз больше, чем AID+RESISC45 (~3.1к) → меньше эпох, чем в ConvNeXtConfig.
    """

    batch_size: int = 32
    max_epochs: int = 20
    loss_eps: float = 1e-4
    patience: int = 3  # early stop: нет роста val macro F1
    min_epochs: int = 3

    lr_candidates: list[float] = field(default_factory=lambda: [1e-3, 1e-4, 1e-5])
    stage1_max_epochs: int = 6
    stage1_min_epochs: int = 2
    stage1_patience: int = 2
    stage2_steps: list[tuple[int, int]] = field(
        default_factory=lambda: [(2, 10), (4, 20)]
    )
    stage2_min_epochs_floor: int = 2
    stage2_max_epochs_per_step: int = 8  # чтобы оба подэтапа успели отработать
    weight_decay: float = 1e-3
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    class_balanced_beta: float = 0.999
    classifier_stage_max_epochs: int = 10
    classifier_stage_min_epochs: int = 3
    classifier_stage_patience: int = 3
    classifier_lr_divisor: int = 3

    n_correct_per_class: int = 2
    figure_size: tuple[int, int] = FIGURE_SIZE

    model_filename: str = "ubc_building_classifier.pth"
    training_curves_plot: str = "ubc_building_training_curves.png"
    confusion_matrix_plot: str = "ubc_building_confusion_matrix.png"
    correct_examples_plot: str = "ubc_building_correct_examples.png"
    error_examples_plot: str = "ubc_building_error_examples.png"


@dataclass(frozen=True)
class NYBuildingConfig:
    """Building-классификатор на Zenodo Building Type (small, ~800 crop'ов).

    Классы сбалансированнее UBC (~420 / ~200 / ~210), но выборка маленькая (США).
    Маски: квадрат по центру из pixle_per_side (CSV Zenodo), формат как UBC (*_mask.png).
    """

    batch_size: int = 16
    max_epochs: int = 40
    loss_eps: float = 1e-4
    patience: int = 4
    min_epochs: int = 5

    lr_candidates: list[float] = field(default_factory=lambda: [1e-3, 1e-4, 1e-5])
    stage1_max_epochs: int = 10
    stage1_min_epochs: int = 3
    stage1_patience: int = 3
    stage2_steps: list[tuple[int, int]] = field(
        default_factory=lambda: [(2, 10), (4, 20)]
    )
    stage2_min_epochs_floor: int = 3
    stage2_max_epochs_per_step: int = 12
    weight_decay: float = 1e-3
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0

    n_correct_per_class: int = 3
    figure_size: tuple[int, int] = FIGURE_SIZE

    model_filename: str = "ny_building_classifier.pth"
    training_curves_plot: str = "ny_building_training_curves.png"
    confusion_matrix_plot: str = "ny_building_confusion_matrix.png"
    correct_examples_plot: str = "ny_building_correct_examples.png"
    error_examples_plot: str = "ny_building_error_examples.png"


@dataclass(frozen=True)
class InriaSegmentationConfig:
    """Бинарная сегментация зданий на патчах INRIA 512×512."""

    batch_size: int = 8
    max_epochs: int = 40
    loss_eps: float = 1e-4
    patience: int = 5
    min_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    pos_weight: float = 2.0

    lr_candidates: list[float] = field(default_factory=lambda: [1e-3, 1e-4, 1e-5])
    stage1_max_epochs: int = 10
    stage1_min_epochs: int = 3
    stage1_patience: int = 3
    stage2_steps: list[tuple[int, int]] = field(
        default_factory=lambda: [(2, 10), (4, 20)]
    )
    stage2_min_epochs_floor: int = 3
    stage2_max_epochs_per_step: int = 12

    inference_patch_size: int = INRIA_PATCH_SIZE
    inference_stride: int = INRIA_PATCH_STRIDE
    n_val_examples: int = 6
    figure_size: tuple[int, int] = FIGURE_SIZE

    model_filename: str = "inria_building_segmenter.pth"
    training_curves_plot: str = "inria_segmentation_curves.png"
    examples_plot: str = "inria_segmentation_examples.png"
    tile_overlay_plot: str = "inria_segmentation_tile_overlay.png"


EDA = EDAConfig()
BASELINE = BaselineConfig()
CONVNEXT_TINY = ConvNeXtConfig()
UBC_BUILDING = UBCBuildingConfig()
NY_BUILDING = NYBuildingConfig()
INRIA_SEGMENTATION = InriaSegmentationConfig()


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Фиксирует seed для random / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    """Создаёт каталоги models/ и reports/, если их нет."""
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

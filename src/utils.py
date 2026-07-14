"""Единая конфигурация проекта (кроме 03_training.ipynb)."""


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


EDA = EDAConfig()
BASELINE = BaselineConfig()
CONVNEXT_TINY = ConvNeXtConfig()
CONVNEXT_XXL = ConvNeXtConfig(
    batch_size=4,
    model_filename="convnext_xxl.pth",
    training_curves_plot="convnext_xxl_training_curves.png",
    confusion_matrix_plot="convnext_xxl_confusion_matrix.png",
    correct_examples_plot="convnext_xxl_correct_examples.png",
    error_examples_plot="convnext_xxl_error_examples.png",
)


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

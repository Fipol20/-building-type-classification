"""Собрать notebooks/colab_ny_building_training.ipynb из 05_ny (без outputs)."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "notebooks" / "05_ny_building_training.ipynb"
OUT = PROJECT_ROOT / "notebooks" / "colab_ny_building_training.ipynb"

ENV_CELL_SOURCE = """# === Среда: локально / Colab / Colab CLI (VDS) ===
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATA_SOURCE = os.environ.get('DATA_SOURCE', 'drive' if Path('/content').exists() else 'local')
os.environ.setdefault('COLAB_ZIP_NAME', 'ny_colab_data.zip')
_COLAB_ROOT = Path('/content/building-type-classification')
_DRIVE_FOLDER = 'building-type-classification'
_ZIP_NAME = os.environ['COLAB_ZIP_NAME']


def _early_colab_unpack() -> None:
    if _COLAB_ROOT.exists():
        shutil.rmtree(_COLAB_ROOT)
    _COLAB_ROOT.mkdir(parents=True)

    my_drive = Path('/content/drive/MyDrive')
    if not my_drive.exists():
        raise FileNotFoundError(
            'Google Drive не смонтирован. На VDS: colab drivemount -s chain04-07'
        )

    drive_project = my_drive / _DRIVE_FOLDER
    zip_path = drive_project / _ZIP_NAME
    if not zip_path.exists():
        root_zip = my_drive / _ZIP_NAME
        if root_zip.exists():
            zip_path = root_zip
    if zip_path.exists():
        print(f'Распаковка {zip_path} ...')
        subprocess.check_call(['unzip', '-q', str(zip_path), '-d', str(_COLAB_ROOT)])
    elif (drive_project / 'data' / 'processed_ny_building').exists():
        print(f'Копирование из {drive_project} ...')
        for item in ['data', 'src', 'requirements.txt', 'models']:
            src = drive_project / item
            if not src.exists():
                continue
            dst = _COLAB_ROOT / item
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    else:
        raise FileNotFoundError(
            f'Нет {_ZIP_NAME} и нет data/processed_ny_building/ в {drive_project}'
        )

    os.chdir(_COLAB_ROOT)


if Path('/content').exists() and DATA_SOURCE == 'drive':
    if not (_COLAB_ROOT / 'src' / 'runtime_env.py').exists():
        _early_colab_unpack()

if Path('/content').exists():
    sys.path.insert(0, str(_COLAB_ROOT / 'src'))
else:
    for _root in (Path.cwd(), Path.cwd().parent):
        if (_root / 'src' / 'runtime_env.py').exists():
            sys.path.insert(0, str((_root / 'src').resolve()))
            break

from runtime_env import (
    colab_pip_install,
    is_colab,
    load_dotenv,
    recommended_batch_size,
    recommended_num_workers,
    setup_notebook_env,
)

PROJECT_ROOT, DRIVE_PROJECT = setup_notebook_env(data_source=DATA_SOURCE if is_colab() else None)
load_dotenv(PROJECT_ROOT)

_hf_token = os.environ.get('HF_TOKEN')
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)
    print('Hugging Face: авторизация OK')

if is_colab():
    colab_pip_install('timm', 'scikit-learn', 'seaborn', 'huggingface_hub')

import torch
LOADER_WORKERS = recommended_num_workers() if is_colab() else 0
print('PROJECT_ROOT:', PROJECT_ROOT)
print('DATA_SOURCE:', DATA_SOURCE)
print('ZIP:', os.environ.get('COLAB_ZIP_NAME'))
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('DataLoader workers:', LOADER_WORKERS)
if is_colab():
    print('Batch (auto):', recommended_batch_size())
"""

MAIN_CELL_SOURCE = """import copy
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import (
    BuildingDataset,
    get_building_train_transforms,
    get_transforms,
    load_split,
)
from model_convnext import build_convnext_tiny, unfreeze_all, unfreeze_stages, count_trainable_params
from utils import (
    NY_BUILDING as CFG,
    NY_BUILDING_CLASSES as CLASSES,
    IMAGE_SIZE,
    MODELS_DIR,
    PLOT_DPI,
    RANDOM_SEED,
    REPORTS_DIR,
    NY_BUILDING_SPLIT_CSV,
    ensure_dirs,
    set_random_seed,
)

BATCH_SIZE = CFG.batch_size
MAX_EPOCHS = CFG.max_epochs
PATIENCE = CFG.patience
MIN_EPOCHS = CFG.min_epochs
WEIGHT_DECAY = CFG.weight_decay
LABEL_SMOOTHING = CFG.label_smoothing
FOCAL_GAMMA = CFG.focal_gamma

LR_CANDIDATES = CFG.lr_candidates
STAGE1_MAX_EPOCHS = CFG.stage1_max_epochs
STAGE1_MIN_EPOCHS = CFG.stage1_min_epochs
STAGE1_PATIENCE = CFG.stage1_patience
STAGE2_STEPS = CFG.stage2_steps
STAGE2_MIN_EPOCHS_FLOOR = CFG.stage2_min_epochs_floor
STAGE2_MAX_EPOCHS_PER_STEP = CFG.stage2_max_epochs_per_step

ensure_dirs()
set_random_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Устройство:', device)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print(f'Focal gamma={FOCAL_GAMMA}, label_smoothing={LABEL_SMOOTHING}, weight_decay={WEIGHT_DECAY}')
print(f'stage2_max_epochs_per_step={STAGE2_MAX_EPOCHS_PER_STEP}')

sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = CFG.figure_size
"""


def _to_nb_source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines if lines else [text]


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    for cell in nb["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None

    nb["cells"][0]["source"] = _to_nb_source(
        """# NY Building-классификатор — Google Colab / VDS

Портативная версия `05_ny_building_training.ipynb` (ConvNeXt-Tiny, 3 класса).

**Результат:** `models/ny_building_classifier.pth` → fine-tune в `06`, пайплайн в `07`/`08`.

## Перед запуском

1. **Данные на Drive:** `MyDrive/building-type-classification/ny_colab_data.zip`
   (на ПК: `python scripts/prepare_colab_ny_upload.py`)
2. **Colab:** Runtime → GPU → Run all
3. **VDS:** `bash /root/scripts/run-chain-04-07.sh` или `run-ny-training.sh`
"""
    )

    env_cell = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": _to_nb_source(ENV_CELL_SOURCE),
    }
    main_cell = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": _to_nb_source(MAIN_CELL_SOURCE),
    }

    # Убираем локальные ячейки HF + imports (индексы 1–2)
    nb["cells"] = [nb["cells"][0], env_cell, main_cell] + nb["cells"][3:]

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "LOADER_WORKERS = 4" in src:
            cell["source"] = _to_nb_source(src.replace("LOADER_WORKERS = 4\n", ""))
        if "best_model_path = MODELS_DIR / CFG.model_filename" in src and "torch.save(stage2_result" in src:
            cell["source"] = _to_nb_source(
                """best_model_path = MODELS_DIR / CFG.model_filename
torch.save(stage2_result['best_state'], best_model_path)
print(f'Модель сохранена: {best_model_path}')

if DRIVE_PROJECT is not None:
    drive_models = DRIVE_PROJECT / 'models'
    drive_models.mkdir(parents=True, exist_ok=True)
    drive_model_path = drive_models / CFG.model_filename
    shutil.copy2(best_model_path, drive_model_path)
    print(f'Копия на Drive: {drive_model_path}')
"""
            )
            break

    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Готово: {OUT} ({len(nb['cells'])} ячеек)")


if __name__ == "__main__":
    main()

"""Собрать notebooks/colab_inria_segmentation.ipynb из 04_inria (без outputs)."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "notebooks" / "04_inria_building_segmentation.ipynb"
OUT = PROJECT_ROOT / "notebooks" / "colab_inria_segmentation.ipynb"

ENV_CELL_SOURCE = """# === Среда: локально / Colab / Colab CLI (VDS) ===
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATA_SOURCE = os.environ.get('DATA_SOURCE', 'drive' if Path('/content').exists() else 'local')
os.environ.setdefault('COLAB_ZIP_NAME', 'inria_colab_data.zip')
_COLAB_ROOT = Path('/content/building-type-classification')
_DRIVE_FOLDER = 'building-type-classification'
_ZIP_NAME = os.environ['COLAB_ZIP_NAME']


def _early_colab_unpack() -> None:
    \"\"\"Распаковка INRIA zip с Drive до import runtime_env (Colab CLI / VDS).\"\"\"
    if _COLAB_ROOT.exists():
        shutil.rmtree(_COLAB_ROOT)
    _COLAB_ROOT.mkdir(parents=True)

    my_drive = Path('/content/drive/MyDrive')
    if not my_drive.exists():
        raise FileNotFoundError(
            'Google Drive не смонтирован. На VDS: colab drivemount -s inria-train'
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
    elif (drive_project / 'data' / 'processed_inria').exists():
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
            f'Нет {_ZIP_NAME} и нет data/processed_inria/ в {drive_project}'
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
    recommended_num_workers,
    recommended_segmentation_batch_size,
    setup_notebook_env,
)

PROJECT_ROOT, DRIVE_PROJECT = setup_notebook_env(data_source=DATA_SOURCE if is_colab() else None)
load_dotenv(PROJECT_ROOT)

if is_colab():
    _seg_dst = PROJECT_ROOT / 'src' / 'model_segmentation.py'
    _seg_dst.write_text({{SEGMENTATION_SRC}}, encoding='utf-8')
    print('model_segmentation.py: записана исправленная версия для timm')
    import sys
    sys.modules.pop('model_segmentation', None)

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
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM: {vram:.1f} GB')
print('DataLoader workers:', LOADER_WORKERS)
if is_colab():
    print('Segmentation batch (auto):', recommended_segmentation_batch_size())
"""

MAIN_CELL_SOURCE = """import copy
import json
import os
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
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

IS_COLAB = is_colab()
if not IS_COLAB:
    sys.path.insert(0, str(Path('..').resolve() / 'src'))
    LOADER_WORKERS = 0

import importlib
import model_segmentation as _model_segmentation
importlib.reload(_model_segmentation)

from dataset import InriaSegmentationDataset, load_inria_split
from model_segmentation import (
    build_convnext_segmenter,
    count_trainable_params,
    unfreeze_encoder_stages,
)
from utils import (
    INRIA_PATCH_SIZE,
    INRIA_PATCH_STRIDE,
    INRIA_RAW_DIR,
    INRIA_SEGMENTATION as CFG,
    INRIA_SPLIT_CSV,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODELS_DIR,
    NUM_WORKERS,
    PLOT_DPI,
    RANDOM_SEED,
    REPORTS_DIR,
    ensure_dirs,
    set_random_seed,
)

BATCH_SIZE = (
    recommended_segmentation_batch_size(CFG.batch_size) if IS_COLAB else CFG.batch_size
)
MAX_EPOCHS = CFG.max_epochs
LOSS_EPS = CFG.loss_eps
PATIENCE = CFG.patience
MIN_EPOCHS = CFG.min_epochs
LEARNING_RATE = CFG.learning_rate
WEIGHT_DECAY = CFG.weight_decay
BCE_WEIGHT = CFG.bce_weight
DICE_WEIGHT = CFG.dice_weight
POS_WEIGHT = CFG.pos_weight

LR_CANDIDATES = CFG.lr_candidates
STAGE1_MAX_EPOCHS = CFG.stage1_max_epochs
STAGE1_MIN_EPOCHS = CFG.stage1_min_epochs
STAGE1_PATIENCE = CFG.stage1_patience
STAGE2_STEPS = CFG.stage2_steps
STAGE2_MIN_EPOCHS_FLOOR = CFG.stage2_min_epochs_floor
STAGE2_MAX_EPOCHS_PER_STEP = CFG.stage2_max_epochs_per_step

INFERENCE_PATCH_SIZE = CFG.inference_patch_size
INFERENCE_STRIDE = CFG.inference_stride
N_VAL_EXAMPLES = CFG.n_val_examples
PRED_THRESHOLD = 0.5

ensure_dirs()
set_random_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Устройство:', device)
print(f'batch_size={BATCH_SIZE} | num_workers={LOADER_WORKERS}')
if torch.cuda.is_available():
    print('ГПУ:', torch.cuda.get_device_name(0))

STAGE1_CKPT = MODELS_DIR / 'inria_segmenter_stage1_best.pth'
STAGE1_META = MODELS_DIR / 'inria_segmenter_stage1_meta.json'
STAGE2_SUB_CKPT = {i: MODELS_DIR / f'inria_segmenter_stage2_sub{i}_best.pth' for i in (1, 2)}
STAGE2_META = MODELS_DIR / 'inria_segmenter_stage2_meta.json'
SKIP_STAGE1_IF_CHECKPOINT = True
SKIP_STAGE2_IF_CHECKPOINT = True

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
        """# INRIA сегментация — Google Colab / VDS

Портативная версия `04_inria_building_segmentation.ipynb` (ConvNeXt-Tiny + FPN, 512×512).

**Результат:** `models/inria_building_segmenter.pth` → `05_zone_building_pipeline.ipynb`

## Перед запуском

1. **Данные на Drive:** `MyDrive/building-type-classification/inria_colab_data.zip`
   (на ПК: `python scripts/prepare_colab_inria_upload.py`)
2. **Colab в браузере:** Runtime → GPU (A100/T4) → Run all
3. **VDS:** OAuth → `bash /root/scripts/run-inria-training.sh` (см. `D:/VDS/README.md`)

Если есть чекпоинт этапа 1 в zip — этап 1 пропустится (`SKIP_STAGE1_IF_CHECKPOINT`).
На VDS только `DATA_SOURCE=drive`. Модель копируется на Drive.
"""
    )

    seg_src = (PROJECT_ROOT / "src" / "model_segmentation.py").read_text(encoding="utf-8")
    env_source = ENV_CELL_SOURCE.replace("{{SEGMENTATION_SRC}}", repr(seg_src))

    env_cell = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": _to_nb_source(env_source),
    }
    nb["cells"].insert(1, env_cell)
    nb["cells"][2]["source"] = _to_nb_source(MAIN_CELL_SOURCE)

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if (
            "checkpoint_path = MODELS_DIR / CFG.model_filename" in src
            and "torch.save(model.state_dict()" in src
        ):
            cell["source"] = _to_nb_source(
                """model = build_convnext_segmenter(pretrained=False, freeze_encoder=False).to(device)
model.load_state_dict(final_state)

checkpoint_path = MODELS_DIR / CFG.model_filename
torch.save(model.state_dict(), checkpoint_path)
print(f'Модель сохранена: {checkpoint_path}')
print(f'Лучший val IoU при обучении: {final_best_iou:.4f}')

if DRIVE_PROJECT is not None:
    drive_models = DRIVE_PROJECT / 'models'
    drive_models.mkdir(parents=True, exist_ok=True)
    drive_model_path = drive_models / CFG.model_filename
    shutil.copy2(checkpoint_path, drive_model_path)
    print(f'Копия на Drive: {drive_model_path}')
"""
            )
            break

    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Готово: {OUT} ({len(nb['cells'])} ячеек)")


if __name__ == "__main__":
    main()

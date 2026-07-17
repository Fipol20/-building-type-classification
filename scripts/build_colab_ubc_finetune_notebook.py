"""Сгенерировать notebooks/colab_ubc_finetune_all.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "notebooks" / "colab_ubc_finetune_all.ipynb"

ENV_CELL = r'''# === Среда: локально / Colab / Colab CLI (VDS) ===
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATA_SOURCE = os.environ.get('DATA_SOURCE', 'drive' if Path('/content').exists() else 'local')
os.environ.setdefault('COLAB_ZIP_NAME', 'ubc_finetune_colab_data.zip')
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
            'Google Drive не смонтирован. На VDS: colab drivemount -s ubc-finetune'
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
    elif (drive_project / 'data' / 'processed_ubc').exists():
        print(f'Копирование из {drive_project} ...')
        for item in ['data', 'src', 'requirements.txt', 'models', 'notebooks']:
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
            f'Нет {_ZIP_NAME} и нет data/processed_ubc/ в {drive_project}'
        )

    os.chdir(_COLAB_ROOT)

    drive_models = drive_project / 'models'
    local_models = _COLAB_ROOT / 'models'
    local_models.mkdir(parents=True, exist_ok=True)
    if drive_models.exists():
        for ckpt in sorted(drive_models.glob('*.pth')):
            shutil.copy2(ckpt, local_models / ckpt.name)
            print('  модель с Drive:', ckpt.name)


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
    recommended_segmentation_batch_size,
    setup_notebook_env,
)

PROJECT_ROOT, DRIVE_PROJECT = setup_notebook_env(data_source=DATA_SOURCE if is_colab() else None)
load_dotenv(PROJECT_ROOT)

if is_colab():
    colab_pip_install('timm', 'scikit-learn', 'seaborn', 'huggingface_hub')

import torch
print('PROJECT_ROOT:', PROJECT_ROOT)
print('DATA_SOURCE:', DATA_SOURCE)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
elif is_colab():
    print('⚠ GPU не найден! Runtime → GPU (T4)')

LOADER_WORKERS = recommended_num_workers()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, '| DataLoader workers:', LOADER_WORKERS)
'''

NY_FINETUNE_CELL = r'''# === 1. Fine-tune NY building на UBC crop'ах ===
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import BuildingDataset, get_building_train_transforms, get_transforms, load_split
from losses import FocalLoss
from model_convnext import build_convnext_tiny, unfreeze_all
from utils import (
    MODELS_DIR,
    NY_BUILDING,
    NY_BUILDING_CLASSES as CLASSES,
    RANDOM_SEED,
    UBC_SPLIT_CSV,
    ensure_dirs,
    set_random_seed,
)

NY_UBC_CKPT = MODELS_DIR / 'ny_building_ubc.pth'
NY_BASE_CKPT = MODELS_DIR / NY_BUILDING.model_filename
NY_FT_EPOCHS = 8
NY_FT_LR = 1e-5
NY_FT_PATIENCE = 3

set_random_seed(RANDOM_SEED)
ensure_dirs()

split_df = load_split(UBC_SPLIT_CSV)
train_df = split_df[split_df['split'] == 'train'].reset_index(drop=True)
train_transform = get_building_train_transforms()
eval_transform = get_transforms()
train_ds = BuildingDataset(split_df, split='train', classes=CLASSES, transform=train_transform, use_mask=True)
val_ds = BuildingDataset(split_df, split='val', classes=CLASSES, transform=eval_transform, use_mask=True)


def make_weighted_sampler(df, classes=CLASSES) -> WeightedRandomSampler:
    class_counts = df['class'].value_counts()
    class_weight = {cls: 1.0 / class_counts[cls] for cls in classes}
    sample_weights = df['class'].map(class_weight).to_numpy()
    return WeightedRandomSampler(sample_weights, num_samples=len(df), replacement=True)


BATCH_SIZE = recommended_batch_size(NY_BUILDING.batch_size)
LOADER_KWARGS = dict(num_workers=LOADER_WORKERS, persistent_workers=LOADER_WORKERS > 0, pin_memory=True)
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, sampler=make_weighted_sampler(train_df, CLASSES), **LOADER_KWARGS
)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **LOADER_KWARGS)
criterion = FocalLoss(gamma=NY_BUILDING.focal_gamma, label_smoothing=NY_BUILDING.label_smoothing)

ny_model = build_convnext_tiny(num_classes=len(CLASSES), freeze_backbone=False)
ny_model.load_state_dict(torch.load(NY_BASE_CKPT, map_location='cpu', weights_only=True))
unfreeze_all(ny_model)
ny_model = ny_model.to(device)

optimizer = torch.optim.AdamW(ny_model.parameters(), lr=NY_FT_LR, weight_decay=NY_BUILDING.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
best_f1, best_state, no_improve = -1.0, None, 0

print(f'NY fine-tune: {len(train_ds)} train / {len(val_ds)} val | base: {NY_BASE_CKPT.name}')


def _ny_epoch(loader, train: bool):
    ny_model.train() if train else ny_model.eval()
    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for imgs, y in loader:
            imgs, y = imgs.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = ny_model(imgs)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(y.cpu().tolist())
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return total_loss / len(loader.dataset), macro_f1


for epoch in range(1, NY_FT_EPOCHS + 1):
    t0 = time.time()
    tr_loss, _ = _ny_epoch(train_loader, train=True)
    va_loss, va_f1 = _ny_epoch(val_loader, train=False)
    scheduler.step(va_f1)
    improved = va_f1 > best_f1
    if improved:
        best_f1, best_state, no_improve = va_f1, copy.deepcopy(ny_model.state_dict()), 0
    else:
        no_improve += 1
    print(
        f'NY эпоха {epoch}/{NY_FT_EPOCHS} | train_loss={tr_loss:.4f} | '
        f'val_loss={va_loss:.4f} macro F1={va_f1:.4f} | {time.time()-t0:.1f}s'
        + (' <- лучшая' if improved else '')
    )
    if epoch >= 2 and no_improve >= NY_FT_PATIENCE:
        print('NY early stop')
        break

if best_state is not None:
    ny_model.load_state_dict(best_state)
torch.save(ny_model.state_dict(), NY_UBC_CKPT)
print(f'Сохранено: {NY_UBC_CKPT} (best val macro F1={best_f1:.4f})')
'''

INRIA_FINETUNE_CELL = r'''# === 2. Fine-tune INRIA segmenter на UBC seg-патчах ===
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import InriaSegmentationDataset, load_inria_split
from model_segmentation import build_convnext_segmenter, unfreeze_encoder_stages
from utils import INRIA_SEGMENTATION, MODELS_DIR, RANDOM_SEED, UBC_SEG_SPLIT_CSV, ensure_dirs, set_random_seed

INRIA_UBC_CKPT = MODELS_DIR / 'inria_building_ubc.pth'
INRIA_BASE_CKPT = MODELS_DIR / INRIA_SEGMENTATION.model_filename
INRIA_FT_EPOCHS = 6
INRIA_FT_LR = 5e-5
INRIA_FT_PATIENCE = 2
PRED_THRESHOLD = 0.5


class DiceLoss(nn.Module):
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = (targets >= 0.5).float()
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * inter + 1) / (union + 1)
        return 1 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, pos_weight: float = INRIA_SEGMENTATION.pos_weight):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)


@torch.no_grad()
def _pixel_iou(logits, targets, threshold=PRED_THRESHOLD):
    preds = (torch.sigmoid(logits) >= threshold).float()
    targets = (targets >= 0.5).float()
    inter = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item() - inter
    return inter / (union + 1e-8)


set_random_seed(RANDOM_SEED)
seg_df = load_inria_split(UBC_SEG_SPLIT_CSV)
train_seg = InriaSegmentationDataset(seg_df, split='train', augment=True)
val_seg = InriaSegmentationDataset(seg_df, split='val', augment=False)

SEG_BATCH = recommended_segmentation_batch_size(INRIA_SEGMENTATION.batch_size)
seg_kwargs = dict(num_workers=LOADER_WORKERS, persistent_workers=LOADER_WORKERS > 0, pin_memory=True)
train_seg_loader = DataLoader(train_seg, batch_size=SEG_BATCH, shuffle=True, **seg_kwargs)
val_seg_loader = DataLoader(val_seg, batch_size=SEG_BATCH, shuffle=False, **seg_kwargs)

seg_model = build_convnext_segmenter(pretrained=False, freeze_encoder=True)
seg_model.load_state_dict(torch.load(INRIA_BASE_CKPT, map_location='cpu', weights_only=True))
unfreeze_encoder_stages(seg_model, n_stages=2)
seg_model = seg_model.to(device)
seg_criterion = SegmentationLoss().to(device)
seg_optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, seg_model.parameters()),
    lr=INRIA_FT_LR,
    weight_decay=INRIA_SEGMENTATION.weight_decay,
)
seg_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(seg_optimizer, mode='max', factor=0.5, patience=2)

best_iou, best_seg_state, seg_no_improve = -1.0, None, 0
print(f'INRIA fine-tune: {len(train_seg)} train / {len(val_seg)} val | base: {INRIA_BASE_CKPT.name}')


def _seg_epoch(loader, train: bool):
    seg_model.train() if train else seg_model.eval()
    total_loss, iou_sum, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            if train:
                seg_optimizer.zero_grad()
            logits = seg_model(images)
            loss = seg_criterion(logits, masks)
            if train:
                loss.backward()
                seg_optimizer.step()
            total_loss += loss.item() * images.size(0)
            iou_sum += _pixel_iou(logits, masks) * images.size(0)
            n += images.size(0)
    return total_loss / n, iou_sum / n


for epoch in range(1, INRIA_FT_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_iou = _seg_epoch(train_seg_loader, train=True)
    va_loss, va_iou = _seg_epoch(val_seg_loader, train=False)
    seg_scheduler.step(va_iou)
    improved = va_iou > best_iou
    if improved:
        best_iou, best_seg_state, seg_no_improve = va_iou, copy.deepcopy(seg_model.state_dict()), 0
    else:
        seg_no_improve += 1
    print(
        f'INRIA эпоха {epoch}/{INRIA_FT_EPOCHS} | train IoU={tr_iou:.4f} | '
        f'val IoU={va_iou:.4f} | {time.time()-t0:.1f}s' + (' <- лучшая' if improved else '')
    )
    if epoch >= 2 and seg_no_improve >= INRIA_FT_PATIENCE:
        print('INRIA early stop')
        break

if best_seg_state is not None:
    seg_model.load_state_dict(best_seg_state)
torch.save(seg_model.state_dict(), INRIA_UBC_CKPT)
print(f'Сохранено: {INRIA_UBC_CKPT} (best val IoU={best_iou:.4f})')
'''

PIPELINE_CELL = r'''# === 3. Калибровка порога + оценка пайплайна на UBC test ===
import pandas as pd
import torch

from model_convnext import build_convnext_tiny
from model_segmentation import build_convnext_segmenter
from pipeline_ubc import (
    evaluate_building_classification,
    run_pipeline_on_tile,
    ubc_tile_names_for_eval_split,
)
from utils import MODELS_DIR, NY_BUILDING, UBC_RAW_DIR

# test в split.csv = исходный UBC val; сырые тайлы лежат в data/raw/ubc/val/
UBC_IMAGE_SPLIT = 'val'

zone_ckpt = MODELS_DIR / 'convnext_best.pth'
zone_model = build_convnext_tiny(num_classes=4, freeze_backbone=False)
zone_model.load_state_dict(torch.load(zone_ckpt, map_location='cpu', weights_only=True))
zone_model = zone_model.to(device).eval()

ny_model.load_state_dict(torch.load(NY_UBC_CKPT, map_location='cpu', weights_only=True))
ny_model = ny_model.to(device).eval()

seg_model.load_state_dict(torch.load(INRIA_UBC_CKPT, map_location='cpu', weights_only=True))
seg_model = seg_model.to(device).eval()

test_tiles = ubc_tile_names_for_eval_split('test')
if not test_tiles:
    test_tiles = sorted(p.name for p in (UBC_RAW_DIR / UBC_IMAGE_SPLIT).glob('*_RGB.tif'))
print(f'Тайлов test (исходный UBC val): {len(test_tiles)}')

THRESHOLDS = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
CALIB_TILES = test_tiles[: min(30, len(test_tiles))]


def _eval_split(tiles, mask_threshold):
    rows = []
    for tile in tiles:
        res = run_pipeline_on_tile(
            tile,
            split=UBC_IMAGE_SPLIT,
            zone_model=zone_model,
            segmenter_model=seg_model,
            building_model=ny_model,
            device=str(device),
            mask_threshold=mask_threshold,
            ubc_raw_dir=UBC_RAW_DIR,
        )
        cls_m = evaluate_building_classification(res)
        rows.append({
            'tile': tile,
            'mask_iou': res.mask_iou,
            'building_macro_f1': cls_m['macro_f1'],
            'building_acc': cls_m['accuracy'],
            'matched': cls_m['matched'],
        })
    return pd.DataFrame(rows)


best_thr, best_score = 0.5, -1.0
for thr in THRESHOLDS:
    df_cal = _eval_split(CALIB_TILES, thr)
    score = df_cal['mask_iou'].mean()
    print(f'  threshold={thr:.2f} → mean mask IoU (calib {len(CALIB_TILES)} tiles) = {score:.4f}')
    if score > best_score:
        best_score, best_thr = score, thr

print(f'\nЛучший порог: {best_thr:.2f}')

test_metrics = _eval_split(test_tiles, best_thr)

print('\n=== TEST (исходный UBC val, n={}) ==='.format(len(test_tiles)))
print(f"mask IoU mean: {test_metrics['mask_iou'].mean():.4f}")
matched = test_metrics[test_metrics['matched'] > 0]
if len(matched):
    print(f"building macro F1: {matched['building_macro_f1'].mean():.4f}")
    print(f"building acc: {matched['building_acc'].mean():.4f}")
else:
    print('building macro F1: n/a (нет сопоставлений)')

print(test_metrics.head(10).to_string())
'''

SAVE_DRIVE_CELL = r'''# === 4. Копирование весов на Google Drive ===
import shutil

if is_colab() and DRIVE_PROJECT is not None:
    DRIVE_PROJECT.mkdir(parents=True, exist_ok=True)
    drive_models = DRIVE_PROJECT / 'models'
    drive_models.mkdir(parents=True, exist_ok=True)
    for ckpt in [NY_UBC_CKPT, INRIA_UBC_CKPT]:
        dst = drive_models / ckpt.name
        shutil.copy2(ckpt, dst)
        print(f'Скопировано на Drive: {dst}')
    print(f'\nГотово. Скачайте из {drive_models} или используйте в 05_zone_building_pipeline.ipynb')
else:
    print('Локальный режим — веса в', MODELS_DIR)
'''


def _code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")],
    }


def _md_cell(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def main() -> None:
    intro = """# UBC Fine-tune + Pipeline — Google Colab

Один ноутбук: **NY building** + **INRIA segmenter** fine-tune на UBC, затем оценка пайплайна `05` на val/test.

**Не использует** `ubc_building_classifier.pth` (06_ubc).

## Перед запуском

1. На ПК: `python scripts/prepare_colab_ubc_finetune.py` → `ubc_finetune_colab_data.zip`
2. Залейте zip на **Google Drive**: `MyDrive/building-type-classification/`
3. Colab: **Runtime → GPU (T4)** → **Run all**

## Результат

- `models/ny_building_ubc.pth`
- `models/inria_building_ubc.pth`
- Метрики пайплайна на UBC val/test (копия весов на Drive)
"""

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": [
            _md_cell(intro, "intro"),
            _code_cell(ENV_CELL, "env"),
            _md_cell("## 1. Fine-tune NY building (ny_building_classifier → ny_building_ubc)", "h-ny"),
            _code_cell(NY_FINETUNE_CELL, "ny"),
            _md_cell("## 2. Fine-tune INRIA segmenter (inria_building_segmenter → inria_building_ubc)", "h-inria"),
            _code_cell(INRIA_FINETUNE_CELL, "inria"),
            _md_cell("## 3. Pipeline eval (zone + ubc-finetuned INRIA + NY, UBC test)", "h-pipe"),
            _code_cell(PIPELINE_CELL, "pipe"),
            _md_cell("## 4. Сохранить на Drive", "h-save"),
            _code_cell(SAVE_DRIVE_CELL, "save"),
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Записано: {OUT}")


if __name__ == "__main__":
    main()

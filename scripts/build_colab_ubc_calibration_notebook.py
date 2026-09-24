"""Сгенерировать notebooks/colab_ubc_pipeline_calibration.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "colab_ubc_pipeline_calibration.ipynb"

ENV_CELL = r'''# === Среда: локально / Colab / Colab CLI (VDS) ===
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATA_SOURCE = os.environ.get('DATA_SOURCE', 'drive' if Path('/content').exists() else 'local')
os.environ.setdefault('COLAB_ZIP_NAME', 'ubc_calib_colab_data.zip')
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
            'Google Drive не смонтирован. На VDS: colab drivemount -s ubc-calib'
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
    elif (drive_project / 'data' / 'processed_ubc' / 'split.csv').exists():
        print(f'Копирование из {drive_project} ...')
        for item in ['data', 'src', 'requirements.txt', 'models', 'notebooks', 'reports']:
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
            f'Нет {_ZIP_NAME} в {drive_project}. Запустите prepare_colab_ubc_calibration.py'
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
    is_headless_colab,
    load_dotenv,
    recommended_batch_size,
    recommended_segmentation_batch_size,
    setup_notebook_env,
)

PROJECT_ROOT, DRIVE_PROJECT = setup_notebook_env(data_source=DATA_SOURCE if is_colab() else None)
load_dotenv(PROJECT_ROOT)

if is_colab():
    colab_pip_install('timm', 'scikit-learn', 'tqdm')

import pandas as pd
import shutil
import torch
from tqdm.auto import tqdm
print('PROJECT_ROOT:', PROJECT_ROOT)
print('DRIVE_PROJECT:', DRIVE_PROJECT)
print('headless:', is_headless_colab())
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
elif is_colab():
    print('⚠ GPU не найден! Runtime → GPU (T4)')

SEG_BATCH = recommended_segmentation_batch_size()
CLS_BATCH = recommended_batch_size(32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, '| seg batch:', SEG_BATCH, '| cls batch:', CLS_BATCH)
'''

CELLS = [
    ("markdown", r"""# UBC Pipeline Calibration — Google Colab / VDS

Совместный подбор порогов **zone → find → class → merge** (~**25–35 мин** на T4/A100).

## Перед запуском (на ПК)

```bash
python scripts/build_colab_ubc_calibration_notebook.py
python scripts/prepare_colab_ubc_calibration.py
```

Залейте **`ubc_calib_colab_data.zip`** на Google Drive:
`MyDrive/building-type-classification/`

## Colab в браузере

1. **Runtime → GPU (T4)**
2. **Run all** — Drive смонтируется автоматически (popup «Разрешить»)

## VDS (автовход — пароль в `.env`, комп можно выключить)

**Один клик с ПК:**

```
D:\VDS\start-ubc-calib.bat
```

Или:

```bash
python D:/VDS/scripts/start_ubc_calib_remote.py
```

Мониторинг: `D:\VDS\watch-ubc-calib.bat`

Google (OAuth + Drive) — **только если сессия `ubc-calib` новая**; один раз через `bash D:/VDS/scripts/vps_ssh.sh`, затем `colab drivemount -s ubc-calib`.

**Test (153 тайла)** — только финальная оценка, без подбора."""),

    ("code", ENV_CELL),

    ("markdown", "## Модели"),

    ("code", """from model_convnext import build_convnext_tiny
from model_segmentation import build_convnext_segmenter
from utils import MODELS_DIR, NY_BUILDING_CLASSES, ZONE_CLASSES

def _pick(primary: str, fallback: str) -> Path:
    p, f = MODELS_DIR / primary, MODELS_DIR / fallback
    if p.exists():
        return p
    if f.exists():
        print(f'  fallback: {fallback}')
        return f
    raise FileNotFoundError(f'Нет {primary} и {fallback}')

zone_ckpt = MODELS_DIR / 'convnext_best.pth'
find_ckpt = _pick('inria_building_ubc.pth', 'inria_building_segmenter.pth')
class_ckpt = _pick('ny_building_ubc.pth', 'ny_building_classifier.pth')

zone_model = build_convnext_tiny(num_classes=len(ZONE_CLASSES), freeze_backbone=False)
zone_model.load_state_dict(torch.load(zone_ckpt, map_location=device, weights_only=True))
zone_model.to(device).eval()

find_model = build_convnext_segmenter(pretrained=False, freeze_encoder=False)
find_model.load_state_dict(torch.load(find_ckpt, map_location=device, weights_only=True))
find_model.to(device).eval()

class_model = build_convnext_tiny(num_classes=len(NY_BUILDING_CLASSES), freeze_backbone=False)
class_model.load_state_dict(torch.load(class_ckpt, map_location=device, weights_only=True))
class_model.to(device).eval()

print('zone:', zone_ckpt.name)
print('find:', find_ckpt.name)
print('class:', class_ckpt.name)"""),

    ("markdown", "## Val-тайлы для калибровки"),

    ("code", """from pipeline_ubc import ubc_tile_names_for_eval_split

CALIB_TILE_LIMIT = 80   # None = все 516 val (дольше)
GRID_MODE = 'colab'     # 'colab' быстрее; 'full' — полная сетка

val_tiles = ubc_tile_names_for_eval_split('val')
if CALIB_TILE_LIMIT:
    val_tiles = val_tiles[:CALIB_TILE_LIMIT]
test_tiles = ubc_tile_names_for_eval_split('test')
print(f'Калибровка: {len(val_tiles)} val-тайлов')
print(f'Test eval:  {len(test_tiles)} тайлов')"""),

    ("markdown", "## Кэш zone + find (один раз на тайл)"),

    ("code", """from pipeline_calibrate import cache_tile_outputs

caches = []
for tile in tqdm(val_tiles, desc='cache val'):
    caches.append(cache_tile_outputs(
        tile,
        zone_model=zone_model,
        find_model=find_model,
        device=str(device),
        seg_batch_size=SEG_BATCH,
    ))
if not caches:
    raise RuntimeError('Кэш пуст — проверьте zip (тайлы, split.csv, модели)')
print('Кэш val:', len(caches))"""),

    ("markdown", """## Калибровка (fast grid)

Classify вызывается **только** для уникальных пар `(mask_threshold, min_area)`.
Пороги confidence и zone — мгновенная фильтрация/merge."""),

    ("code", """import time

from pipeline_calibrate import PipelineParams, calibrate

t0 = time.perf_counter()
result = calibrate(
    caches,
    base_params=PipelineParams(),
    class_model=class_model,
    device=str(device),
    refine=True,
    fast=True,
    grid_mode=GRID_MODE,
    class_batch_size=CLS_BATCH,
)
elapsed = time.perf_counter() - t0

print(f'Калибровка за {elapsed/60:.1f} мин')
print('=== Лучшие параметры ===')
for k, v in result.best_params.to_dict().items():
    print(f'  {k}: {v}')
print()
print('=== Метрики val ===')
for k, v in result.best_metrics.__dict__.items():
    print(f'  {k}: {v}')

import pandas as pd
display(pd.DataFrame(result.top_results).head(10))
if result.best_metrics.n_tiles == 0 or result.best_metrics.joint_score == 0:
    print('⚠ joint=0 или n_tiles=0 — кэш/данные не загрузились, параметры НЕ валидны')"""),

    ("markdown", "## Сохранение на Drive"),

    ("code", """from pipeline_calibrate import DEFAULT_PARAMS_PATH
from utils import REPORTS_DIR, ensure_dirs

ensure_dirs()
params_path = result.best_params.save_json(DEFAULT_PARAMS_PATH)
print('Локально:', params_path)

if DRIVE_PROJECT is not None:
    drive_reports = DRIVE_PROJECT / 'reports'
    drive_reports.mkdir(parents=True, exist_ok=True)
    drive_params = drive_reports / 'pipeline_calibrated_params.json'
    shutil.copy2(params_path, drive_params)
    print('Drive:', drive_params)

    top_csv = REPORTS_DIR / 'pipeline_calibration_top.csv'
    pd.DataFrame(result.top_results).to_csv(top_csv, index=False)
    shutil.copy2(top_csv, drive_reports / 'pipeline_calibration_top.csv')
    print('Top-10 CSV на Drive')"""),

    ("markdown", "## Финальная оценка на TEST"),

    ("code", """from pipeline_calibrate import evaluate_params, run_pipeline_from_cache
from pipeline_ubc import evaluate_building_classification

best = result.best_params

test_caches = []
for tile in tqdm(test_tiles, desc='cache test'):
    test_caches.append(cache_tile_outputs(
        tile,
        zone_model=zone_model,
        find_model=find_model,
        device=str(device),
        seg_batch_size=SEG_BATCH,
    ))

test_metrics = evaluate_params(test_caches, best, class_model, str(device))
print('=== TEST (frozen params) ===')
for k, v in test_metrics.__dict__.items():
    print(f'  {k}: {v}')

rows = []
for cache in test_caches:
    r = run_pipeline_from_cache(cache, best, class_model, str(device))
    cls = evaluate_building_classification(r, iou_threshold=best.match_iou)
    rows.append({
        'tile': cache.tile_name,
        'mask_iou': r.mask_iou,
        'building_macro_f1': cls['macro_f1'],
        'building_acc': cls['accuracy'],
        'matched': cls['matched'],
    })
test_df = pd.DataFrame(rows)
test_csv = REPORTS_DIR / 'pipeline_calibrated_test_metrics.csv'
test_df.to_csv(test_csv, index=False)
print('CSV:', test_csv)

if DRIVE_PROJECT is not None:
    shutil.copy2(test_csv, DRIVE_PROJECT / 'reports' / test_csv.name)
    print('Test CSV на Drive')

display(test_df.head(10))"""),
]


def _cell(cell_type: str, source: str, cid: str) -> dict:
    if cell_type == "markdown":
        return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": [l + "\n" for l in source.split("\n")]}
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in source.split("\n")],
    }


def main() -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [_cell(t, s, f"c{i}") for i, (t, s) in enumerate(CELLS)],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Записано: {OUT}")


if __name__ == "__main__":
    main()

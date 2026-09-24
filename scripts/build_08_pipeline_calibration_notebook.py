"""Сгенерировать notebooks/08_pipeline_calibration.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "08_pipeline_calibration.ipynb"

CELLS = [
    ("markdown", """# Совместная калибровка пайплайна (zone + find + class + merge)

Подбор порогов **на val-тайлах** (516 шт., `raw/ubc/train/` + `raw/ubc/val/`) — модели их не видели как целые снимки при fine-tune.

**Test (153 тайла)** используется только в финальной оценке.

Joint score = 0.4×mask IoU + 0.5×building macro F1 + 0.1×resolved residential rate"""),

    ("code", """import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path('..').resolve() / 'src'))

from model_convnext import build_convnext_tiny
from model_segmentation import build_convnext_segmenter
from pipeline_calibrate import (
    DEFAULT_PARAMS_PATH,
    PipelineParams,
    cache_tile_outputs,
    calibrate,
    evaluate_params,
    run_pipeline_from_cache,
)
from pipeline_ubc import evaluate_building_classification, ubc_tile_names_for_eval_split
from runtime_env import recommended_batch_size, recommended_segmentation_batch_size
from utils import MODELS_DIR, NY_BUILDING_CLASSES, REPORTS_DIR, ensure_dirs

ensure_dirs()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEG_BATCH = recommended_segmentation_batch_size()
CLS_BATCH = recommended_batch_size(32)
print('device:', device, '| seg batch:', SEG_BATCH)"""),

    ("markdown", "## Модели"),

    ("code", """def _pick(primary: str, fallback: str) -> Path:
    p, f = MODELS_DIR / primary, MODELS_DIR / fallback
    return p if p.exists() else f

zone_ckpt = MODELS_DIR / 'convnext_best.pth'
find_ckpt = _pick('inria_building_ubc.pth', 'inria_building_segmenter.pth')
class_ckpt = _pick('ny_building_ubc.pth', 'ny_building_classifier.pth')

zone_model = build_convnext_tiny(num_classes=4, freeze_backbone=False)
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

    ("code", """CALIB_TILE_LIMIT = 80   # None = все 516; 80 для быстрого прогона
GRID_MODE = 'colab'       # 'colab' (~30 мин) | 'full' (дольше, точнее)

val_tiles = ubc_tile_names_for_eval_split('val')
if CALIB_TILE_LIMIT:
    val_tiles = val_tiles[:CALIB_TILE_LIMIT]
print(f'Калибровка на {len(val_tiles)} val-тайлах (raw/train + raw/val, авто)')

test_tiles = ubc_tile_names_for_eval_split('test')
print(f'Test для финала: {len(test_tiles)} тайлов (raw/val/)')"""),

    ("markdown", "## Кэш предсказаний (INRIA prob_map + zone prob_map)"),

    ("code", """caches = []
for tile in tqdm(val_tiles, desc='cache val'):
    caches.append(cache_tile_outputs(
        tile, zone_model=zone_model, find_model=find_model, device=str(device),
        seg_batch_size=SEG_BATCH,
    ))
print('Кэш готов:', len(caches))"""),

    ("markdown", "## Калибровка (fast: coarse + refine)"),

    ("code", """import time

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
print(f'Калибровка за {(time.perf_counter()-t0)/60:.1f} мин')

print('=== Лучшие параметры ===')
for k, v in result.best_params.to_dict().items():
    print(f'  {k}: {v}')
print()
print('=== Метрики на val ===')
for k, v in result.best_metrics.__dict__.items():
    print(f'  {k}: {v}')

top_df = pd.DataFrame(result.top_results)
display(top_df.head(10))"""),

    ("markdown", "## Сохранение параметров"),

    ("code", """params_path = result.best_params.save_json(DEFAULT_PARAMS_PATH)
print('Сохранено:', params_path)"""),

    ("markdown", "## Финальная оценка на TEST (frozen params, без подбора)"),

    ("code", """best = result.best_params

# кэш test (тяжёлые модели — один раз)
test_caches = []
for tile in tqdm(test_tiles, desc='cache test'):
    test_caches.append(cache_tile_outputs(
        tile, zone_model=zone_model, find_model=find_model, device=str(device),
        seg_batch_size=SEG_BATCH,
    ))

test_metrics = evaluate_params(test_caches, best, class_model, str(device))
print('=== TEST (153 тайла, params frozen) ===')
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
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Записано: {OUT}")


if __name__ == "__main__":
    main()

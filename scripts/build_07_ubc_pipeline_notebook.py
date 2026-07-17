"""Сгенерировать notebooks/07_ubc_four_stage_pipeline.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "07_ubc_four_stage_pipeline.ipynb"

CELLS = [
    ("markdown", """# UBC test: 4 этапа на каждом тайле

Только **test**-тайлы (`split=test` в `processed_ubc/split.csv`) — исходный UBC **val**, 153 снимка.
Эти тайлы **не использовались** при fine-tune NY и INRIA (обучение было на train-тайлах).

На **каждом** test-изображении по очереди:

| Этап | Модель | Что делает |
|------|--------|------------|
| **1. Zone** | `convnext_best.pth` | карта застройки (dense/sparse/commercial/industrial) |
| **2. Find** | `inria_building_ubc.pth` | бинарная маска зданий → instances (bbox) |
| **3. Class** | `ny_building_ubc.pth` | тип каждого здания (residential/commercial/industrial) |
| **4. Merge** | правило слияния | residential + zone → dense/sparse; итоговая карта |

Модели 2–3: UBC-finetuned (если есть в `models/`), иначе базовые.

**Пороги:** из `reports/pipeline_calibrated_params.json` (ноутбук 08), иначе дефолты.

**Визуализация:** для каждого test-тайла — сетка 4×2 (pred | GT), легенда цветов, PNG в `reports/four_stage_test/`."""),

    ("code", """import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path('..').resolve() / 'src'))

from building_masks import Building, draw_building_masks
from inria_inference import mask_to_buildings, rasterize_buildings_mask
from merge_maps import MERGED_COLORS, draw_merged_masks, merge_summary
from model_convnext import build_convnext_tiny
from model_segmentation import build_convnext_segmenter
from pipeline_calibrate import DEFAULT_PARAMS_PATH, PipelineParams
from pipeline_ubc import evaluate_building_classification, run_pipeline_on_tile, ubc_tile_names_for_eval_split, zone_overlay
from utils import (
    BUILDING_COLORS,
    MODELS_DIR,
    NY_BUILDING_CLASSES,
    PLOT_DPI,
    REPORTS_DIR,
    UBC_RAW_DIR,
    ZONE_CLASSES,
    ZONE_COLORS,
    ensure_dirs,
)

ensure_dirs()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Устройство:', device)"""),

    ("markdown", "## Загрузка 3 моделей"),

    ("code", """def _pick_model(primary: str, fallback: str) -> Path:
    p, f = MODELS_DIR / primary, MODELS_DIR / fallback
    if p.exists():
        return p
    if f.exists():
        print(f'  fallback: {fallback}')
        return f
    raise FileNotFoundError(f'Нет {primary} и {fallback}')

zone_ckpt = _pick_model('convnext_best.pth', 'convnext_best.pth')
find_ckpt = _pick_model('inria_building_ubc.pth', 'inria_building_segmenter.pth')
class_ckpt = _pick_model('ny_building_ubc.pth', 'ny_building_classifier.pth')

zone_model = build_convnext_tiny(num_classes=len(ZONE_CLASSES), freeze_backbone=False)
zone_model.load_state_dict(torch.load(zone_ckpt, map_location=device, weights_only=True))
zone_model.to(device).eval()

find_model = build_convnext_segmenter(pretrained=False, freeze_encoder=False)
find_model.load_state_dict(torch.load(find_ckpt, map_location=device, weights_only=True))
find_model.to(device).eval()

class_model = build_convnext_tiny(num_classes=len(NY_BUILDING_CLASSES), freeze_backbone=False)
class_model.load_state_dict(torch.load(class_ckpt, map_location=device, weights_only=True))
class_model.to(device).eval()

print('1 Zone:  ', zone_ckpt.name)
print('2 Find:  ', find_ckpt.name)
print('3 Class: ', class_ckpt.name)"""),

    ("markdown", """## Test-тайлы (не из fine-tune train)

`split=test` = исходный UBC val. Сырые файлы: `data/raw/ubc/val/*.tif`."""),

    ("code", """PIPELINE_PARAMS = PipelineParams.load_json(DEFAULT_PARAMS_PATH) or PipelineParams()
if DEFAULT_PARAMS_PATH.exists():
    print('Параметры из', DEFAULT_PARAMS_PATH)
    print('  (подобраны на val, test не участвовал в калибровке)')
else:
    print('Калибровка не найдена — дефолтные пороги. Запустите 08_pipeline_calibration.ipynb')
print(PIPELINE_PARAMS.to_dict())

UBC_IMAGE_SPLIT = 'val'
EVAL_SPLIT = 'test'
MATCH_IOU = PIPELINE_PARAMS.match_iou
SAVE_ALL_VIS = True          # сохранить PNG для каждого test-тайла
SHOW_EACH_IN_NOTEBOOK = False  # True = plt.show() на всех 153 (очень долго)
VIS_DIR = REPORTS_DIR / 'four_stage_test'

test_tiles = ubc_tile_names_for_eval_split(EVAL_SPLIT)
if not test_tiles:
    test_tiles = sorted(p.name for p in (UBC_RAW_DIR / UBC_IMAGE_SPLIT).glob('*_RGB.tif'))

print(f'Test-тайлов (не train): {len(test_tiles)}')
print('Визуализации:', VIS_DIR)"""),

    ("markdown", """## Легенда цветов

### Этап 1 — Zone (карта застройки)
| Цвет | Класс |
|------|-------|
| 🔴 красный | commercial |
| 🟠 оранжевый | industrial |
| 🔵 синий | dense_residential |
| 🔵 голубой | sparse_residential |

### Этап 2 — Find (маска зданий)
| Цвет | Значение |
|------|----------|
| 🔴 красный полупрозрачный | предсказанная маска здания |
| 🔴 красный (GT) | истинная маска из COCO |

### Этап 3 — Class (тип здания, 3 класса)
| Цвет | Класс |
|------|-------|
| 🔵 голубой | residential |
| 🔴 красный | commercial |
| 🟠 оранжевый | industrial |

### Этап 4 — Merge (итог)
Те же цвета, что zone + **нерешённый residential** (голубой как у residential), если zone не дала dense/sparse.

**GT для merge:** в UBC нет разметки dense/sparse — справа показан **GT класс здания** (3 класса)."""),

    ("code", """from matplotlib.patches import Patch

ZONE_LABELS_RU = {
    'commercial': 'commercial (коммерция)',
    'industrial': 'industrial (промышленность)',
    'dense_residential': 'dense_residential (плотная застройка)',
    'sparse_residential': 'sparse_residential (редкая застройка)',
}
BUILDING_LABELS_RU = {
    'residential': 'residential (жилое)',
    'commercial': 'commercial (коммерция)',
    'industrial': 'industrial (промышленность)',
}
MERGE_LABELS_RU = {**ZONE_LABELS_RU, 'residential': 'residential нерешённый (merge)'}

def _legend_figure(title: str, colors: dict, labels: dict) -> plt.Figure:
    patches = [Patch(facecolor=np.array(colors[k]) / 255, label=labels.get(k, k)) for k in colors]
    fig, ax = plt.subplots(figsize=(8, max(1.5, 0.35 * len(patches))))
    ax.legend(handles=patches, loc='center', frameon=False, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    return fig

fig1 = _legend_figure('Этап 1 — Zone', ZONE_COLORS, ZONE_LABELS_RU)
fig2 = _legend_figure('Этап 3 — Class (и GT)', BUILDING_COLORS, BUILDING_LABELS_RU)
fig3 = _legend_figure('Этап 4 — Merge', MERGED_COLORS, MERGE_LABELS_RU)
plt.show()"""),

    ("markdown", """## Визуализация: pred | GT для каждого этапа

Сетка **4×2** на тайл: слева предсказание, справа истина (GT).
Этап 1: GT zone в UBC **нет** — справа исходный снимок."""),

    ("code", """def _mask_overlay(image: Image.Image, mask: np.ndarray, color=(255, 80, 80)) -> Image.Image:
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask.astype(bool)] = (*color, 140)
    return Image.alpha_composite(image.convert('RGBA'), Image.fromarray(overlay, 'RGBA')).convert('RGB')


def _gt_mask_overlay(image: Image.Image, gt_buildings: list[Building]) -> Image.Image:
    gt_mask = rasterize_buildings_mask(gt_buildings, image.size)
    return _mask_overlay(image, gt_mask, color=(220, 40, 40))


def visualize_four_stages_with_gt(result: dict, save_dir: Path | None = None, show: bool = False) -> Path | None:
    \"\"\"4 строки × 2 колонки: pred | GT на каждом этапе.\"\"\"
    image = result['image']
    gt = result['gt_buildings']

    rows = [
        (zone_overlay(image, result['zone_map']), image.copy(), '1. Zone', 'исходник (GT zone нет)'),
        (_mask_overlay(image, result['pred_mask']), _gt_mask_overlay(image, gt), '2. Find', 'GT маска'),
        (draw_building_masks(image, result['buildings_class'], use_predicted=True),
         draw_building_masks(image, gt, use_predicted=False), '3. Class', 'GT класс'),
        (draw_merged_masks(image, result['buildings_merge']),
         draw_building_masks(image, gt, use_predicted=False), '4. Merge', 'GT класс (3 cls)'),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 20))
    for row, (pred_img, gt_img, stage, gt_lbl) in enumerate(rows):
        axes[row, 0].imshow(pred_img)
        axes[row, 0].set_title(f'{stage} — pred', fontsize=11)
        axes[row, 0].axis('off')
        axes[row, 1].imshow(gt_img)
        axes[row, 1].set_title(f'{stage} — {gt_lbl}', fontsize=11)
        axes[row, 1].axis('off')

    cls = evaluate_building_classification(result['pipeline_result'], iou_threshold=MATCH_IOU)
    fig.suptitle(
        f\"{result['tile_name']}\\n\"
        f\"mask IoU={result['mask_iou']:.3f} | building F1={cls['macro_f1']:.3f} | \"
        f\"find={len(result['buildings_find'])} gt={len(gt)}\",
        fontsize=10,
    )
    plt.tight_layout()

    out_path = None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f\"{Path(result['tile_name']).stem}_pred_gt.png\"
        plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


@torch.no_grad()
def process_test_tile_four_stages(tile_name: str) -> dict:
    result = run_pipeline_on_tile(
        tile_name,
        split=UBC_IMAGE_SPLIT,
        zone_model=zone_model,
        segmenter_model=find_model,
        building_model=class_model,
        device=str(device),
        pipeline_params=PIPELINE_PARAMS,
    )
    from copy import deepcopy
    buildings_after_find = mask_to_buildings(
        result.pred_mask, min_area=PIPELINE_PARAMS.min_component_area
    )
    buildings_after_class = deepcopy(result.buildings)
    for b in buildings_after_class:
        b.final_class = None
    return {
        'tile_name': tile_name,
        'image': result.image,
        'zone_map': result.zone_map,
        'prob_map': result.prob_map,
        'pred_mask': result.pred_mask,
        'buildings_find': buildings_after_find,
        'buildings_class': [b for b in result.buildings if b.pred_class is not None],
        'buildings_merge': [b for b in result.buildings if b.final_class is not None],
        'gt_buildings': result.gt_buildings,
        'mask_iou': result.mask_iou,
        'pipeline_result': result,
    }"""),

    ("markdown", """## Весь test-сплит: метрики + визуализация pred|GT

Один проход по всем test-тайлам: метрики в CSV, PNG в `reports/four_stage_test/`."""),

    ("code", """rows = []
saved_paths = []

for tile_name in tqdm(test_tiles, desc='test tiles'):
    r = process_test_tile_four_stages(tile_name)
    cls = evaluate_building_classification(r['pipeline_result'], iou_threshold=MATCH_IOU)
    rows.append({
        'tile': tile_name,
        'n_find': len(r['buildings_find']),
        'n_gt': len(r['gt_buildings']),
        'mask_iou': r['mask_iou'],
        'matched': cls['matched'],
        'building_acc': cls['accuracy'],
        'building_macro_f1': cls['macro_f1'],
        'merge_dense': merge_summary(r['buildings_merge']).get('dense_residential', 0),
        'merge_sparse': merge_summary(r['buildings_merge']).get('sparse_residential', 0),
        'merge_commercial': merge_summary(r['buildings_merge']).get('commercial', 0),
        'merge_industrial': merge_summary(r['buildings_merge']).get('industrial', 0),
        'merge_unresolved': merge_summary(r['buildings_merge']).get('residential', 0),
    })
    if SAVE_ALL_VIS:
        p = visualize_four_stages_with_gt(
            r, save_dir=VIS_DIR, show=SHOW_EACH_IN_NOTEBOOK
        )
        if p:
            saved_paths.append(p)

metrics_df = pd.DataFrame(rows)
matched = metrics_df[metrics_df['matched'] > 0]
print(f'=== TEST n={len(metrics_df)} (не из fine-tune train) ===')
print(f'mask IoU mean:     {metrics_df["mask_iou"].mean():.4f}')
if len(matched):
    print(f'building macro F1: {matched["building_macro_f1"].mean():.4f}')
    print(f'building acc:      {matched["building_acc"].mean():.4f}')
print(metrics_df.head(10).to_string())

out_csv = REPORTS_DIR / 'ubc_test_four_stage_metrics.csv'
metrics_df.to_csv(out_csv, index=False)
print('CSV:', out_csv)
print(f'PNG сохранено: {len(saved_paths)} → {VIS_DIR}')"""),

    ("markdown", """## Сводка по этапам

| Этап | Метрика |
|------|---------|
| 2 Find | mask IoU vs GT |
| 3 Class | macro F1 (matched bbox) |
| 4 Merge | распределение final_class |"""),

    ("code", """print('--- Этап 2 Find: mask IoU ---')
print(metrics_df['mask_iou'].describe()[['mean', 'std', 'min', '50%', 'max']].to_string())
print()
print('--- Этап 3 Class (тайлы с match) ---')
if len(matched):
    print(matched[['building_macro_f1', 'building_acc']].describe().loc[['mean', 'std']].to_string())
print()
print('--- Этап 4 Merge: сумма по датасету ---')
for col in ['merge_dense', 'merge_sparse', 'merge_commercial', 'merge_industrial', 'merge_unresolved']:
    print(f'  {col}: {metrics_df[col].sum()}')"""),
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
    cells = [_cell(t, s, f"c{i}") for i, (t, s) in enumerate(CELLS)]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Записано: {OUT}")


if __name__ == "__main__":
    main()

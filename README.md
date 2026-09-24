# Классификация застройки по аэроснимкам

По спутниковому или аэрофотоснимку определяет тип застройки и классифицирует отдельные здания.

Пайплайн из четырёх этапов (ноутбук `08`):

1. **Zone** — карта застройки по тайлу (ConvNeXt-Tiny, 4 класса, sliding window)
2. **Find** — бинарная маска зданий (сегментатор INRIA, sliding window)
3. **Class** — тип каждого здания (классификатор NY, 3 класса)
4. **Merge** — `residential` уточняется до dense/sparse по zone-карте (центроид)

## Классы

Карта застройки (zone):

| Класс | Смысл |
|---|---|
| `commercial` | коммерческая |
| `industrial` | промышленная |
| `dense_residential` | плотная жилая |
| `sparse_residential` | редкая жилая |

Отдельные здания: `residential`, `commercial`, `industrial`.

## Метрики

| Этап | Модель | Метрика |
|---|---|---|
| Baseline zone | SimpleCNN | macro F1 0.96 |
| Zone | ConvNeXt-Tiny | macro F1 0.99, accuracy 0.99 |
| Find после fine-tune на UBC | INRIA segmenter | val IoU 0.76 |
| Сквозной пайплайн на 153 тайлах UBC | все этапы | mask IoU 0.53, building macro F1 0.26 |

Слабый сквозной F1 — это не скрытый результат, а ограничение переноса: zone и building обучались на AID/RESISC45 и NY, затем частично адаптировались на UBC. Подробности — в разделе «Ограничения».

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Опционально: `HF_TOKEN` в `.env` в корне проекта — быстрее скачивание весов `timm`.

Данные в git не лежат. Их нужно скачать и подготовить скриптами из `scripts/` (запускать из корня проекта):

```bash
python scripts/download_datasets.py
python scripts/download_extra_datasets.py --inria --ubc-crops
python scripts/prepare_inria_patches.py
python scripts/build_inria_split.py
python scripts/prepare_ny_building_dataset.py
python scripts/build_ny_building_split.py
python scripts/download_ubc_raw.py
python scripts/build_ubc_split.py
python scripts/prepare_ubc_seg_patches.py
python scripts/build_ubc_seg_split.py
```

Ожидаемые каталоги после подготовки:

- `data/processed/` — AID + RESISC45 для zone
- `data/processed_inria/` — патчи 512×512 INRIA
- `data/processed_ny_building/` — кропы зданий NY / Zenodo
- `data/processed_ubc/` — кропы зданий UBC + маски
- `data/processed_ubc_seg/` — seg-патчи UBC для fine-tune INRIA
- `data/raw/ubc/`, `data/raw/inria/`, `data/raw/NY_type_small/` — исходники

Веса моделей (`models/*.pth`) тоже не в git: их обучают ноутбуки ниже.

## Ноутбуки

Идти по порядку:

| # | Файл | Что делает | Результат |
|---|---|---|---|
| 01 | `notebooks/01_eda.ipynb` | EDA | — |
| 02 | `notebooks/02_baseline.ipynb` | SimpleCNN baseline | `models/baseline_model.pth` |
| 03 | `notebooks/03_convnext_training.ipynb` | zone-классификатор | `models/convnext_best.pth` |
| 04 | `notebooks/04_inria_building_segmentation.ipynb` | сегментация зданий | `models/inria_building_segmenter.pth` |
| 05 | `notebooks/05_ny_building_training.ipynb` | building-классификатор | `models/ny_building_classifier.pth` |
| 06 | `notebooks/06_ubc_finetune_all.ipynb` | fine-tune NY + INRIA на UBC | `models/ny_building_ubc.pth`, `models/inria_building_ubc.pth` |
| 07 | `notebooks/07_ubc_pipeline_calibration.ipynb` | подбор порогов пайплайна | `reports/pipeline_calibrated_params.json` |
| 08 | `notebooks/08_zone_building_pipeline.ipynb` | финальный пайплайн на UBC | `reports/full_pipeline/` |

## Демо

```bash
streamlit run app.py
```

Два режима: полный пайплайн из четырёх этапов или классификация одного кропа здания. Инференс идёт через `src/predict.py` и пороги из `reports/pipeline_calibrated_params.json`.

## Структура

```
app.py          Streamlit UI
src/            dataset, модели, zone_map, inria_inference,
                building_masks, merge_maps, pipeline_ubc, predict
scripts/        скачивание и подготовка данных
notebooks/      ноутбуки 01–08
models/         веса (обучаются ноутбуками, не в git)
reports/        графики, метрики, калибровка порогов
data/           датасеты (локально, не в git)
```

Метрики в отчётах: accuracy, F1, macro F1, mask IoU, confusion matrix.

## Ограничения

- Калибровка (`07`) и оценка (`08`) — на UBC. Перенос на другие города требует fine-tune или новой калибровки.
- Building-классификатор обучен на кропах США (Zenodo); UBC fine-tune (`06`) адаптирует его только частично. Отсюда низкий сквозной building F1.
- Карта zone зональная (sliding window), не кадастровая точность.
- `dense` / `sparse` для residential считается по zone в центроиде здания — грубое приближение на смешанных участках.
- Residential может остаться «нерешённым» в merge, если zone в точке центроида — commercial или industrial.

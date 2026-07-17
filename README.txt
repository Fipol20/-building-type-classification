Building Type Classification
============================

Задача: по спутниковому/аэрофото снимку определить тип застройки и классифицировать
отдельные здания.

Пайплайн (4 этапа, ноутбук 08):
  1. Zone   — карта застройки по тайлу (ConvNeXt-Tiny, 4 класса, sliding window)
  2. Find   — бинарная маска зданий (INRIA segmenter, sliding window)
  3. Class  — тип каждого здания (NY building classifier, 3 класса)
  4. Merge  — residential уточняется до dense/sparse по zone-карте (центроид)

Классы zone (карта застройки, 4 класса):
  commercial          - коммерческая
  industrial          - промышленная
  dense_residential   - плотная жилая
  sparse_residential  - редкая жилая

Классы building (отдельные здания, 3 класса):
  residential
  commercial
  industrial

Данные (локально, не в git):
  data/processed/              - AID + RESISC45 для zone (03)
                                 split: data/processed/split.csv
  data/processed_inria/        - патчи 512×512 INRIA (04)
                                 split: data/processed_inria/split.csv
  data/processed_ny_building/  - crop'ы зданий NY small / Zenodo (05)
                                 split: data/processed_ny_building/split.csv
  data/processed_ubc/          - crop'ы зданий UBC + *_mask.png (06, 07, 08)
                                 split: data/processed_ubc/split.csv
  data/processed_ubc_seg/      - seg-патчи UBC для fine-tune INRIA (06)
                                 split: data/processed_ubc_seg/split.csv
  data/raw/ubc/                - тайлы UBC (TIF) + COCO (use_coarse_*.json)
  data/raw/inria/              - исходные тайлы INRIA (опционально, viz в 04)
  data/raw/NY_type_small/      - Zenodo Building Type small (для 05)

Установка:
  python -m venv .venv
  .venv\Scripts\activate          # Windows
  pip install -r requirements.txt

  Опционально: HF_TOKEN в .env (корень проекта) — быстрее скачивание весов timm.

Ноутбуки (по порядку):
  01  notebooks/01_eda.ipynb                         - EDA
  02  notebooks/02_baseline.ipynb                    - SimpleCNN baseline
  03  notebooks/03_convnext_training.ipynb             - zone-классификатор
                                                      -> models/convnext_best.pth
  04  notebooks/04_inria_building_segmentation.ipynb  - сегментация зданий (INRIA)
                                                      -> models/inria_building_segmenter.pth
  05  notebooks/05_ny_building_training.ipynb          - building-классификатор (NY)
                                                      -> models/ny_building_classifier.pth
  06  notebooks/06_ubc_finetune_all.ipynb              - fine-tune NY + INRIA на UBC
                                                      -> models/ny_building_ubc.pth
                                                         models/inria_building_ubc.pth
  07  notebooks/07_ubc_pipeline_calibration.ipynb    - подбор порогов пайплайна
                                                      -> reports/pipeline_calibrated_params.json
  08  notebooks/08_zone_building_pipeline.ipynb        - финальный пайплайн на UBC val/test
                                                      -> reports/full_pipeline/

Подготовка данных (скрипты в scripts/, запускать с корня проекта):
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

Модели (models/, не в git — обучаются ноутбуками):
  convnext_best.pth              - zone (03)
  inria_building_segmenter.pth   - find, базовая (04)
  ny_building_classifier.pth     - class, базовая (05)
  ny_building_ubc.pth            - class после UBC fine-tune (06)
  inria_building_ubc.pth         - find после UBC fine-tune (06)

Streamlit (inference на загруженном изображении):
  streamlit run app.py

  Режимы: полный пайплайн (4 этапа) или классификация одного crop'а здания.
  Использует src/predict.py и reports/pipeline_calibrated_params.json.

Структура:
  data/       - датасеты и split.csv
  notebooks/  - ноутбуки 01–08
  src/        - dataset, модели, zone_map, inria_inference, building_masks,
                merge_maps, pipeline_ubc, pipeline_calibrate, predict
  scripts/    - скачивание и подготовка данных
  models/     - веса (.pth)
  reports/    - графики, метрики, pipeline_calibrated_params.json
  app.py      - Streamlit UI

Метрики: accuracy, F1, macro F1, mask IoU, confusion matrix.

Ограничения:
  - Калибровка (07) и оценка (08) — на UBC; перенос на другие города требует
    fine-tune или новой калибровки.
  - NY building обучен на crop'ах США (Zenodo); UBC fine-tune (06) частично
    адаптирует классификатор.
  - Карта zone — зональная (sliding window), не кадастровая точность.
  - dense/sparse для residential — по zone в центроиде здания, грубое приближение
    на смешанных участках.
  - Residential может остаться «нерешённым» в merge, если zone в точке центроида —
    commercial/industrial.

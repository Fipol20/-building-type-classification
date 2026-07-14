Building Type Classification
============================

Задача: по снимку определить тип застройки (4 класса).

Классы:
  commercial          - коммерческая
  industrial          - промышленная
  dense_residential   - плотная жилая
  sparse_residential  - редкая жилая

Данные: AID + RESISC45, 3139 фото.
Split: train 2197 / val 471 / test 471 (файл data/processed/split.csv).

Установка:
  pip install -r requirements.txt

Ноутбуки (по порядку):
  notebooks/01_eda.ipynb           - проверка данных
  notebooks/02_baseline.ipynb        - SimpleCNN
  notebooks/03_training.ipynb        - ResNet18
  notebooks/04_convnext_training.ipynb - ConvNeXt-Tiny
Результаты обучения:
  models/   - веса
  reports/  - графики, матрица ошибок

Структура:
  data/       - датасеты и split.csv
  notebooks/  - ноутбуки
  src/        - код датасета и моделей
  scripts/    - split
  models/     - сохранённые веса
  reports/    - графики

Метрики: accuracy, F1, macro F1, confusion matrix.

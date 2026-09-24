"""Упаковка UBC fine-tune + pipeline eval для Google Colab.

Создаёт ubc_finetune_colab_data.zip с:
  - data/processed_ubc/          (crop'ы + split.csv для NY fine-tune)
  - data/processed_ubc_seg/      (патчи 512 + split.csv для INRIA fine-tune)
  - data/raw/ubc/val, test, annotations  (тайлы для оценки пайплайна)
  - src/
  - requirements.txt
  - models/convnext_best.pth, ny_building_classifier.pth, inria_building_segmenter.pth
  - notebooks/colab_ubc_finetune_all.ipynb

Перед упаковкой (если нет seg-патчей):
  python scripts/prepare_ubc_seg_patches.py
  python scripts/build_ubc_seg_split.py

Использование:
    python scripts/build_colab_ubc_finetune_notebook.py   # сначала ноутбук
    python scripts/prepare_colab_ubc_finetune.py

Результат: ubc_finetune_colab_data.zip в корне репозитория (~1–1.5 ГБ).
Залейте на Google Drive: MyDrive/building-type-classification/
Затем откройте notebooks/colab_ubc_finetune_all.ipynb в Colab (GPU), Run all.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = PROJECT_ROOT / "ubc_finetune_colab_data.zip"

INCLUDE_DIRS = [
    PROJECT_ROOT / "data" / "processed_ubc",
    PROJECT_ROOT / "data" / "processed_ubc_seg",
    PROJECT_ROOT / "src",
]
INCLUDE_RAW_UBC = [
    PROJECT_ROOT / "data" / "raw" / "ubc" / "val",
    PROJECT_ROOT / "data" / "raw" / "ubc" / "annotations",
]
INCLUDE_FILES = [
    PROJECT_ROOT / "requirements.txt",
]
REQUIRED_MODELS = [
    PROJECT_ROOT / "models" / "convnext_best.pth",
    PROJECT_ROOT / "models" / "ny_building_classifier.pth",
    PROJECT_ROOT / "models" / "inria_building_segmenter.pth",
]
OPTIONAL_NOTEBOOK = PROJECT_ROOT / "notebooks" / "colab_ubc_finetune_all.ipynb"


def _ensure_ubc_seg_data() -> None:
    seg_split = PROJECT_ROOT / "data" / "processed_ubc_seg" / "split.csv"
    if seg_split.exists():
        return
    print("Нет processed_ubc_seg — запускаю подготовку патчей...")
    subprocess.check_call([sys.executable, str(PROJECT_ROOT / "scripts" / "prepare_ubc_seg_patches.py")])
    subprocess.check_call([sys.executable, str(PROJECT_ROOT / "scripts" / "build_ubc_seg_split.py")])


def _add_tree(zf: zipfile.ZipFile, directory: Path) -> None:
    if not directory.exists():
        raise FileNotFoundError(f"Папка не найдена: {directory}")
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            arcname = path.relative_to(PROJECT_ROOT).as_posix()
            zf.write(path, arcname)
            print(f"  + {arcname}")


def main() -> None:
    ubc_split = PROJECT_ROOT / "data" / "processed_ubc" / "split.csv"
    if not ubc_split.exists():
        raise FileNotFoundError(
            f"{ubc_split} не найден. Сначала подготовьте UBC crop'ы (processed_ubc)."
        )

    _ensure_ubc_seg_data()

    for model_path in REQUIRED_MODELS:
        if not model_path.exists():
            raise FileNotFoundError(f"Нет базовой модели: {model_path}")

    if not OPTIONAL_NOTEBOOK.exists():
        build_script = PROJECT_ROOT / "scripts" / "build_colab_ubc_finetune_notebook.py"
        if build_script.exists():
            print("Ноутбук не найден — генерирую...")
            subprocess.check_call([sys.executable, str(build_script)])
        if not OPTIONAL_NOTEBOOK.exists():
            raise FileNotFoundError(
                f"{OPTIONAL_NOTEBOOK} не найден. Запустите build_colab_ubc_finetune_notebook.py"
            )

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    print("=== Упаковка ubc_finetune_colab_data.zip ===")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for directory in INCLUDE_DIRS:
            _add_tree(zf, directory)

        for directory in INCLUDE_RAW_UBC:
            _add_tree(zf, directory)

        for file_path in INCLUDE_FILES:
            arcname = file_path.relative_to(PROJECT_ROOT).as_posix()
            zf.write(file_path, arcname)
            print(f"  + {arcname}")

        for model_path in REQUIRED_MODELS:
            arcname = model_path.relative_to(PROJECT_ROOT).as_posix()
            zf.write(model_path, arcname)
            print(f"  + {arcname} (модель)")

        arcname = OPTIONAL_NOTEBOOK.relative_to(PROJECT_ROOT).as_posix()
        zf.write(OPTIONAL_NOTEBOOK, arcname)
        print(f"  + {arcname} (ноутбук)")

    size_gb = OUTPUT_ZIP.stat().st_size / (1024**3)
    print(f"\nГотово: {OUTPUT_ZIP}")
    print(f"Размер: {size_gb:.2f} ГБ")
    print("\nДальше:")
    print("  1. Залейте ubc_finetune_colab_data.zip на Google Drive")
    print("     в папку MyDrive/building-type-classification/")
    print("  2. Colab: notebooks/colab_ubc_finetune_all.ipynb, GPU, Run all")
    print("  3. Результат на Drive: ny_building_ubc.pth, inria_building_ubc.pth")


if __name__ == "__main__":
    main()

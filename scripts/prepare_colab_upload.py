"""Упаковка файлов для загрузки на Google Drive / Colab.

Создаёт building_colab_data.zip с:
  - data/processed_ubc/  (crop'ы + split.csv)
  - src/                 (код проекта)
  - requirements.txt

Использование:
    python scripts/prepare_colab_upload.py

Результат: building_colab_data.zip в корне репозитория (~280 МБ).
Залейте zip на Google Drive в папку building-type-classification/,
затем откройте notebooks/colab_building_training.ipynb в Colab.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = PROJECT_ROOT / "building_colab_data.zip"

INCLUDE_DIRS = [
    PROJECT_ROOT / "data" / "processed_ubc",
    PROJECT_ROOT / "src",
]
INCLUDE_FILES = [
    PROJECT_ROOT / "requirements.txt",
]


def main() -> None:
    split_csv = PROJECT_ROOT / "data" / "processed_ubc" / "split.csv"
    if not split_csv.exists():
        raise FileNotFoundError(
            f"{split_csv} не найден. Сначала подготовьте данные UBC."
        )

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for directory in INCLUDE_DIRS:
            if not directory.exists():
                raise FileNotFoundError(f"Папка не найдена: {directory}")
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    arcname = path.relative_to(PROJECT_ROOT).as_posix()
                    zf.write(path, arcname)
                    print(f"  + {arcname}")

        for file_path in INCLUDE_FILES:
            if not file_path.exists():
                raise FileNotFoundError(f"Файл не найден: {file_path}")
            arcname = file_path.relative_to(PROJECT_ROOT).as_posix()
            zf.write(file_path, arcname)
            print(f"  + {arcname}")

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\nГотово: {OUTPUT_ZIP}")
    print(f"Размер: {size_mb:.1f} МБ")
    print("\nДальше:")
    print("  1. Залейте building_colab_data.zip на Google Drive")
    print("     в папку MyDrive/building-type-classification/")
    print("  2. Colab в браузере: notebooks/colab_building_training.ipynb, GPU, Run all")
    print("  3. VDS: scp notebooks/colab_building_training.ipynb root@VPS:/root/")
    print("     затем bash /root/scripts/run-training.sh")


if __name__ == "__main__":
    main()

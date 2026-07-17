"""Упаковка INRIA-сегментации для Google Drive / Colab.

Создаёт inria_colab_data.zip с:
  - data/processed_inria/   (патчи + split.csv)
  - src/                    (код проекта)
  - requirements.txt
  - models/inria_segmenter_stage1_best.pth  (если есть)
  - models/inria_segmenter_stage1_meta.json (если есть)

Использование:
    python scripts/prepare_colab_inria_upload.py

Результат: inria_colab_data.zip в корне репозитория (~5.5 ГБ).
Залейте zip на Google Drive в папку building-type-classification/,
затем откройте notebooks/04_inria_building_segmentation.ipynb в Colab (GPU).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = PROJECT_ROOT / "inria_colab_data.zip"

INCLUDE_DIRS = [
    PROJECT_ROOT / "data" / "processed_inria",
    PROJECT_ROOT / "src",
]
INCLUDE_FILES = [
    PROJECT_ROOT / "requirements.txt",
]
OPTIONAL_CHECKPOINTS = [
    PROJECT_ROOT / "models" / "inria_segmenter_stage1_best.pth",
    PROJECT_ROOT / "models" / "inria_segmenter_stage1_meta.json",
]
OPTIONAL_NOTEBOOK = PROJECT_ROOT / "notebooks" / "colab_inria_segmentation.ipynb"


def main() -> None:
    split_csv = PROJECT_ROOT / "data" / "processed_inria" / "split.csv"
    if not split_csv.exists():
        raise FileNotFoundError(
            f"{split_csv} не найден. Сначала:\n"
            "  python scripts/prepare_inria_patches.py\n"
            "  python scripts/build_inria_split.py"
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

        for file_path in OPTIONAL_CHECKPOINTS:
            if file_path.exists():
                arcname = file_path.relative_to(PROJECT_ROOT).as_posix()
                zf.write(file_path, arcname)
                print(f"  + {arcname} (чекпоинт)")
            else:
                print(f"  - {file_path.name} (нет — этап 1 ещё не сохранён)")

        if OPTIONAL_NOTEBOOK.exists():
            arcname = OPTIONAL_NOTEBOOK.relative_to(PROJECT_ROOT).as_posix()
            zf.write(OPTIONAL_NOTEBOOK, arcname)
            print(f"  + {arcname} (ноутбук)")

    size_gb = OUTPUT_ZIP.stat().st_size / (1024**3)
    print(f"\nГотово: {OUTPUT_ZIP}")
    print(f"Размер: {size_gb:.2f} ГБ")
    print("\nДальше:")
    print("  1. Залейте inria_colab_data.zip на Google Drive")
    print("     в папку MyDrive/building-type-classification/")
    print("  2. Colab в браузере: notebooks/colab_inria_segmentation.ipynb, GPU, Run all")
    print("  3. VDS: bash D:/VDS/scripts/upload-from-pc.sh inria")
    print("         bash /root/scripts/run-inria-training.sh")
    print("  Этап 1 пропустится, если чекпоинт stage1 есть в zip.")


if __name__ == "__main__":
    main()

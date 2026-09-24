"""Упаковка UBC pipeline calibration для Google Colab.

Создаёт ubc_calib_colab_data.zip (~200–400 МБ):
  - data/processed_ubc/split.csv
  - data/raw/ubc/annotations/
  - data/raw/ubc/{train,val}/ — только тайлы для калибровки (80 val) + test (153)
  - src/, requirements.txt
  - models/ (zone + find + class, UBC-finetuned если есть)
  - notebooks/colab_ubc_pipeline_calibration.ipynb

Использование:
    python scripts/build_colab_ubc_calibration_notebook.py
    python scripts/prepare_colab_ubc_calibration.py

Залейте zip на Google Drive: MyDrive/building-type-classification/
Colab: GPU → Run all (калибровка ~30–45 мин на T4).
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = PROJECT_ROOT / "ubc_calib_colab_data.zip"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline_ubc import resolve_ubc_raw_split, ubc_tile_names_for_eval_split  # noqa: E402

CALIB_TILE_LIMIT = 80
INCLUDE_FILES = [
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "data" / "processed_ubc" / "split.csv",
]
REQUIRED_MODELS = [
    PROJECT_ROOT / "models" / "convnext_best.pth",
]
OPTIONAL_MODELS = [
    PROJECT_ROOT / "models" / "inria_building_ubc.pth",
    PROJECT_ROOT / "models" / "inria_building_segmenter.pth",
    PROJECT_ROOT / "models" / "ny_building_ubc.pth",
    PROJECT_ROOT / "models" / "ny_building_classifier.pth",
]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "colab_ubc_pipeline_calibration.ipynb"


def _required_tile_names() -> set[str]:
    val = ubc_tile_names_for_eval_split("val")
    if CALIB_TILE_LIMIT:
        val = val[:CALIB_TILE_LIMIT]
    test = ubc_tile_names_for_eval_split("test")
    return set(val) | set(test)


def _add_file(zf: zipfile.ZipFile, path: Path, arcname: str | None = None) -> None:
    arc = arcname or path.relative_to(PROJECT_ROOT).as_posix()
    zf.write(path, arc)
    print(f"  + {arc}")


def main() -> None:
    if not NOTEBOOK.exists():
        build = PROJECT_ROOT / "scripts" / "build_colab_ubc_calibration_notebook.py"
        subprocess.check_call([sys.executable, str(build)])

    tile_names = _required_tile_names()
    if not tile_names:
        raise FileNotFoundError("Нет тайлов в processed_ubc/split.csv")

    for model_path in REQUIRED_MODELS:
        if not model_path.exists():
            raise FileNotFoundError(f"Нет модели: {model_path}")

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    print(f"=== ubc_calib_colab_data.zip ({len(tile_names)} тайлов) ===")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for directory in [PROJECT_ROOT / "src", PROJECT_ROOT / "data" / "raw" / "ubc" / "annotations"]:
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    _add_file(zf, path)

        for tile in sorted(tile_names):
            split = resolve_ubc_raw_split(tile)
            path = PROJECT_ROOT / "data" / "raw" / "ubc" / split / tile
            if not path.exists():
                raise FileNotFoundError(f"Нет тайла: {path}")
            _add_file(zf, path)

        for file_path in INCLUDE_FILES:
            _add_file(zf, file_path)

        for model_path in REQUIRED_MODELS + OPTIONAL_MODELS:
            if model_path.exists():
                _add_file(zf, model_path)

        _add_file(zf, NOTEBOOK)

    size_mb = OUTPUT_ZIP.stat().st_size / (1024**2)
    print(f"\nГотово: {OUTPUT_ZIP} ({size_mb:.0f} МБ)")
    print("\nДальше:")
    print("  1. Залейте ubc_calib_colab_data.zip на Google Drive")
    print("     → MyDrive/building-type-classification/")
    print("  2. Colab: colab_ubc_pipeline_calibration.ipynb, Runtime → GPU, Run all")
    print("  3. Результат: reports/pipeline_calibrated_params.json (+ копия на Drive)")


if __name__ == "__main__":
    main()

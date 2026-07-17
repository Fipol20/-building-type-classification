"""Упаковка NY building training для Google Drive / Colab."""

from __future__ import annotations

import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = PROJECT_ROOT / "ny_colab_data.zip"

INCLUDE_DIRS = [
    PROJECT_ROOT / "data" / "processed_ny_building",
    PROJECT_ROOT / "src",
]
INCLUDE_FILES = [PROJECT_ROOT / "requirements.txt"]
OPTIONAL_CHECKPOINTS = [
    PROJECT_ROOT / "models" / "ny_building_stage1_best.pth",
    PROJECT_ROOT / "models" / "ny_building_stage1_meta.json",
]


def main() -> None:
    split_csv = PROJECT_ROOT / "data" / "processed_ny_building" / "split.csv"
    if not split_csv.exists():
        raise FileNotFoundError(
            f"{split_csv} не найден. Сначала:\n"
            "  python scripts/prepare_ny_building_dataset.py\n"
            "  python scripts/build_ny_building_split.py"
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
            zf.write(file_path, file_path.relative_to(PROJECT_ROOT).as_posix())
            print(f"  + {file_path.name}")

        for file_path in OPTIONAL_CHECKPOINTS:
            if file_path.exists():
                arcname = file_path.relative_to(PROJECT_ROOT).as_posix()
                zf.write(file_path, arcname)
                print(f"  + {arcname} (чекпоинт)")

    size_mb = OUTPUT_ZIP.stat().st_size / 1e6
    print(f"\nГотово: {OUTPUT_ZIP} ({size_mb:.0f} МБ)")
    print("Залейте на Drive: MyDrive/building-type-classification/ny_colab_data.zip")


if __name__ == "__main__":
    main()

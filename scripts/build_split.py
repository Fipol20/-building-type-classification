"""Создание объединённого split.csv из AID + RESISC45."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import build_combined_dataframe, dataset_summary, make_split, save_split
from utils import DATA_ROOT, SPLIT_CSV


def main() -> None:
    df = build_combined_dataframe(DATA_ROOT)
    print(f"Всего изображений: {len(df)}")
    print("\nПо датасетам и классам:")
    print(dataset_summary(df).to_string(index=False))

    split_df = make_split(df)
    save_split(split_df, SPLIT_CSV)

    print(f"\nРазбиение:")
    print(split_df.groupby("split").size())
    print(f"\nСохранено: {SPLIT_CSV}")


if __name__ == "__main__":
    main()

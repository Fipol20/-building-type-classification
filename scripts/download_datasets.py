"""Скачивание RESISC45 — только нужные классы.

AID уже лежит в data/processed/{class}/ и копируется в data/aid/.
"""

from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RESISC45_MAP = {
    "commercial_area": "commercial",
    "industrial_area": "industrial",
    "dense_residential": "dense_residential",
    "sparse_residential": "sparse_residential",
}


def download_resisc45(out_dir: Path) -> dict[str, int]:
    """Скачивает 4 класса RESISC45 через HuggingFace (timm/resisc45)."""
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    for target in RESISC45_MAP.values():
        (out_dir / target).mkdir(exist_ok=True)

    print("Загрузка RESISC45 (timm/resisc45)...")
    ds = load_dataset("timm/resisc45", split="train")
    label_names = ds.features["label"].names

    counts: dict[str, int] = {v: 0 for v in RESISC45_MAP.values()}
    for i, row in enumerate(ds):
        label = label_names[row["label"]]
        if label not in RESISC45_MAP:
            continue
        target = RESISC45_MAP[label]
        dst = out_dir / target / f"resisc45_{i:05d}.jpg"
        if not dst.exists():
            row["image"].convert("RGB").save(dst, quality=95)
        counts[target] += 1

    return counts


def setup_aid(aid_dir: Path, processed_dir: Path) -> dict[str, int]:
    """Копирует AID из data/processed в data/aid (если ещё нет)."""
    aid_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for cls_dir in processed_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        if cls_dir.name not in {"commercial", "industrial", "dense_residential", "sparse_residential"}:
            continue
        dst_dir = aid_dir / cls_dir.name
        dst_dir.mkdir(exist_ok=True)
        n = 0
        for img in cls_dir.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            dst = dst_dir / f"aid_{img.name}"
            if not dst.exists():
                shutil.copy2(img, dst)
            n += 1
        counts[cls_dir.name] = n
    return counts


def main() -> None:
    processed = DATA_DIR / "processed"
    aid_dir = DATA_DIR / "aid"
    resisc_dir = DATA_DIR / "resisc45"

    print("=== AID (из data/processed) ===")
    aid_counts = setup_aid(aid_dir, processed)
    for cls, n in sorted(aid_counts.items()):
        print(f"  {cls}: {n}")

    print("\n=== RESISC45 ===")
    resisc_counts = download_resisc45(resisc_dir)
    for cls, n in sorted(resisc_counts.items()):
        print(f"  {cls}: {n}")

    print("\nГотово. Затем: python scripts/build_split.py")


if __name__ == "__main__":
    main()

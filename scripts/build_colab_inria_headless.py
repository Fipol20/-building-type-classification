"""Собрать headless-ноутбук INRIA: только этап 2 + сохранение модели (VDS)."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_NB = PROJECT_ROOT / "notebooks" / "colab_inria_segmentation.ipynb"
OUT_NB = PROJECT_ROOT / "notebooks" / "colab_inria_headless.ipynb"

INLINE_UNFREEZE = '''
def _unfreeze_encoder_stages_inline(model, n_stages: int) -> None:
    """Colab/timm features_only: stages_0..3 (без .stages)."""
    def _set(m, rg):
        for p in m.parameters():
            p.requires_grad = rg
    _set(model, False)
    n = max(1, min(int(n_stages), 4))
    stages = [getattr(model.encoder, f"stages_{i}") for i in range(4)]
    for st in stages[-n:]:
        _set(st, True)
    _set(model.decoder, True)

unfreeze_encoder_stages = _unfreeze_encoder_stages_inline
'''

SKIP_CELL = "print('Пропуск визуализации (headless / VDS)')"

SAVE_CELL = """checkpoint_path = MODELS_DIR / CFG.model_filename
if 'final_state' not in globals() or final_state is None:
    print('final_state нет — сохраняем лучший stage1 чекпоинт')
    final_state = torch.load(STAGE1_CKPT, map_location='cpu', weights_only=True)
    final_best_iou = json.loads(STAGE1_META.read_text(encoding='utf-8'))['best_iou']

model = build_convnext_segmenter(pretrained=False, freeze_encoder=False).to(device)
model.load_state_dict(final_state)
torch.save(model.state_dict(), checkpoint_path)
print(f'Модель сохранена: {checkpoint_path}')
print(f'Лучший val IoU: {final_best_iou:.4f}')

if DRIVE_PROJECT is not None:
    drive_models = DRIVE_PROJECT / 'models'
    drive_models.mkdir(parents=True, exist_ok=True)
    drive_model_path = drive_models / CFG.model_filename
    shutil.copy2(checkpoint_path, drive_model_path)
    print(f'Копия на Drive: {drive_model_path}')
"""


def main() -> None:
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    new_cells = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "def run_progressive_finetune" in src and "unfreeze_encoder_stages(model, n_stages)" in src:
            cell["source"] = [INLINE_UNFREEZE + "\n" + src]
            cell["outputs"] = []
            cell["execution_count"] = None
            new_cells.append(cell)
            continue
        if any(
            x in src
            for x in (
                "training_history['train_loss']",
                "predict_building_mask_sliding",
                "tile_pred.shape",
                "austin_dir = INRIA_RAW_DIR",
                "predict_masks(model, val_loader",
                "for split_name, loader in",
            )
        ):
            cell = {
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
                "source": [SKIP_CELL],
            }
        if (
            "checkpoint_path = MODELS_DIR / CFG.model_filename" in src
            and "model.load_state_dict(final_state)" in src
        ):
            cell["source"] = [SAVE_CELL]
            cell["outputs"] = []
            cell["execution_count"] = None
        if cell.get("cell_type") == "markdown" and any(
            h in src for h in ("Кривые", "Визуализация", "Инференс", "Демо:", "Итоги")
        ):
            continue
        new_cells.append(cell)

    nb["cells"] = new_cells
    nb["cells"][0]["source"] = [
        "# INRIA headless — только этап 2 + сохранение (VDS)\n\n"
        "Без графиков и raw-тайлов. Этап 1 из чекпоинта в zip.\n"
    ]
    OUT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Готово: {OUT_NB} ({len(new_cells)} ячеек)")


if __name__ == "__main__":
    main()

"""Обучение building-классификатора (3 класса UBC: residential/commercial/industrial).

Дисбаланс UBC crop'ов (примерно):
  residential ~33.5k | commercial ~4.9k | industrial ~0.76k (~1.9%).
Industrial мало в самой разметке use_coarse (не ошибка сплита);
public/other (~14k) намеренно не используются.

Стратегия (decoupling + class-balanced focal, Kang et al. / Cui et al.):
  Фаза A — представление (backbone): естественное распределение train,
    ClassBalancedFocalLoss (без WeightedRandomSampler), прогрессивная разморозка.
  Фаза B — классификатор (head): backbone заморожен, WeightedRandomSampler +
    обычный FocalLoss на сбалансированных батчах.

Запуск:
    python scripts/train_ubc_building_classifier.py

Предпосылка: data/processed_ubc/split.csv (создаётся scripts/build_ubc_split.py).
Результат: models/ubc_building_classifier.pth + графики/матрица в reports/.
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import (  # noqa: E402
    BuildingDataset,
    get_building_train_transforms,
    get_transforms,
    load_split,
)
from losses import ClassBalancedFocalLoss, FocalLoss, compute_class_balanced_alpha  # noqa: E402
from model_convnext import (  # noqa: E402
    build_convnext_tiny,
    count_trainable_params,
    freeze_backbone_only,
    unfreeze_all,
    unfreeze_stages,
)
from utils import (  # noqa: E402
    IMAGE_SIZE,
    MODELS_DIR,
    NUM_WORKERS,
    PLOT_DPI,
    RANDOM_SEED,
    REPORTS_DIR,
    UBC_BUILDING as CFG,
    UBC_CLASSES as CLASSES,
    UBC_SPLIT_CSV,
    ensure_dirs,
    set_random_seed,
)

BATCH_SIZE = CFG.batch_size
MAX_EPOCHS = CFG.max_epochs
PATIENCE = CFG.patience
MIN_EPOCHS = CFG.min_epochs
WEIGHT_DECAY = CFG.weight_decay
LABEL_SMOOTHING = CFG.label_smoothing
FOCAL_GAMMA = CFG.focal_gamma
CLASS_BALANCED_BETA = CFG.class_balanced_beta

LR_CANDIDATES = CFG.lr_candidates
STAGE1_MAX_EPOCHS = CFG.stage1_max_epochs
STAGE1_MIN_EPOCHS = CFG.stage1_min_epochs
STAGE1_PATIENCE = CFG.stage1_patience
STAGE2_STEPS = CFG.stage2_steps
STAGE2_MIN_EPOCHS_FLOOR = CFG.stage2_min_epochs_floor
STAGE2_MAX_EPOCHS_PER_STEP = CFG.stage2_max_epochs_per_step

CLASSIFIER_MAX_EPOCHS = CFG.classifier_stage_max_epochs
CLASSIFIER_MIN_EPOCHS = CFG.classifier_stage_min_epochs
CLASSIFIER_PATIENCE = CFG.classifier_stage_patience
CLASSIFIER_LR_DIVISOR = CFG.classifier_lr_divisor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_counts_from_df(df, classes: list[str] = CLASSES) -> dict[str, int]:
    counts = df["class"].value_counts()
    return {cls: int(counts[cls]) for cls in classes}


def make_weighted_sampler(df, classes: list[str] = CLASSES) -> WeightedRandomSampler:
    """Каждый класс равновероятен в батче (фаза B — decoupled classifier)."""
    class_counts = df["class"].value_counts()
    class_weight = {cls: 1.0 / class_counts[cls] for cls in classes}
    sample_weights = df["class"].map(class_weight).to_numpy()
    return WeightedRandomSampler(sample_weights, num_samples=len(df), replacement=True)


def make_class_balanced_criterion(class_counts: dict[str, int]) -> ClassBalancedFocalLoss:
    alpha = compute_class_balanced_alpha(class_counts, CLASSES, beta=CLASS_BALANCED_BETA)
    return ClassBalancedFocalLoss(
        alpha=alpha, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING
    )


def make_focal_criterion() -> FocalLoss:
    return FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)


def run_epoch(loader, model, criterion, optimizer=None):
    """Один проход: обучение или оценка."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, accuracy, macro_f1


@torch.no_grad()
def collect_predictions(loader, model):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_loop(
    model,
    train_loader,
    val_loader,
    lr,
    criterion,
    max_epochs=MAX_EPOCHS,
    min_epochs=MIN_EPOCHS,
    patience=PATIENCE,
    weight_decay=WEIGHT_DECAY,
    label="",
    verbose=True,
):
    """Цикл обучения с ранней остановкой по отсутствию роста val macro F1."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_f1", "lr"]}
    best_val_f1, best_epoch = -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improve = 0
    prefix = f"[{label}] " if label else ""

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc, _ = run_epoch(train_loader, model, criterion, optimizer)
        val_loss, val_acc, val_f1 = run_epoch(val_loader, model, criterion, optimizer=None)
        scheduler.step(val_f1)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(lr_now)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if verbose:
            marker = " <- лучшая" if is_best else ""
            print(
                f"{prefix}эпоха {epoch:3d}/{max_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} macro F1={val_f1:.4f} | "
                f"{time.time() - t0:.1f}s{marker}"
            )

        if epoch >= min_epochs and epochs_without_improve >= patience:
            if verbose:
                print(f"{prefix}early stop: val macro F1 не растёт {patience} эпох")
            break

    return {
        "history": history,
        "best_f1": best_val_f1,
        "best_acc": history["val_acc"][best_epoch - 1],
        "best_epoch": best_epoch,
        "best_state": best_state,
        "epochs_run": len(history["train_loss"]),
    }


def run_stage1(train_loader, val_loader, criterion):
    """Поиск LR с замороженным backbone (обучается только head)."""
    best_state, best_f1, best_lr = None, -1.0, None
    results = {}

    for lr in LR_CANDIDATES:
        torch.manual_seed(RANDOM_SEED)
        model = build_convnext_tiny(num_classes=len(CLASSES), freeze_backbone=True).to(device)
        print(f"\n--- Этап 1 | LR={lr:.0e} | обучаемых параметров: {count_trainable_params(model):,} ---")

        result = train_loop(
            model, train_loader, val_loader, lr, criterion,
            max_epochs=STAGE1_MAX_EPOCHS,
            min_epochs=STAGE1_MIN_EPOCHS,
            patience=STAGE1_PATIENCE,
            label=f"этап1_{lr:.0e}",
        )
        results[lr] = result

        if result["best_f1"] > best_f1:
            best_f1, best_state, best_lr = result["best_f1"], result["best_state"], lr

    print(f"\nЛучший LR этапа 1: {best_lr:.0e} (macro F1={best_f1:.4f})")
    return best_state, best_lr, best_f1, results


def run_stage2(train_loader, val_loader, criterion, stage1_state, best_lr):
    """Прогрессивная разморозка; лимит эпох на подэтап, чтобы оба шага отработали."""
    combined_history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_f1", "lr"]}
    best_f1, best_acc, best_state, best_substage = -1.0, 0.0, None, None
    remaining_epochs = MAX_EPOCHS
    carry_state = stage1_state
    carry_lr = None
    substage_results = []

    for step_idx, (n_stages, lr_divisor) in enumerate(STAGE2_STEPS):
        if remaining_epochs <= 0:
            break

        torch.manual_seed(RANDOM_SEED)
        model = build_convnext_tiny(num_classes=len(CLASSES), freeze_backbone=True).to(device)
        model.load_state_dict(carry_state)

        if step_idx == len(STAGE2_STEPS) - 1:
            unfreeze_all(model)
            stage_label = "все слои"
        else:
            unfreeze_stages(model, n_stages)
            stage_label = f"последние {n_stages} стадии"

        target_lr = best_lr / lr_divisor
        fine_tune_lr = min(carry_lr, target_lr) if carry_lr is not None else target_lr
        step_max = min(STAGE2_MAX_EPOCHS_PER_STEP, remaining_epochs)
        print(
            f"\n--- Этап 2, шаг {step_idx + 1}/{len(STAGE2_STEPS)}: {stage_label} | "
            f"LR={fine_tune_lr:.1e} | max_epochs={step_max} | "
            f"обучаемых параметров: {count_trainable_params(model):,} ---"
        )

        result = train_loop(
            model, train_loader, val_loader, fine_tune_lr, criterion,
            max_epochs=step_max,
            min_epochs=max(STAGE2_MIN_EPOCHS_FLOOR, MIN_EPOCHS // 3),
            label=f"этап2_{step_idx + 1}",
        )

        for key in combined_history:
            combined_history[key].extend(result["history"][key])

        substage_results.append(
            {
                "substage": step_idx + 1,
                "epochs_run": result["epochs_run"],
                "best_f1": result["best_f1"],
                "best_epoch": result["best_epoch"],
            }
        )

        remaining_epochs = MAX_EPOCHS - len(combined_history["train_loss"])
        carry_state = result["best_state"]
        carry_lr = result["history"]["lr"][-1]

        if result["best_f1"] > best_f1:
            best_f1, best_acc, best_state, best_substage = (
                result["best_f1"], result["best_acc"], result["best_state"], step_idx + 1,
            )

    return {
        "history": combined_history,
        "best_f1": best_f1,
        "best_acc": best_acc,
        "best_state": best_state,
        "epochs_run": len(combined_history["train_loss"]),
        "substage_results": substage_results,
        "best_substage": best_substage,
    }


def run_representation_phase(train_loader, val_loader, criterion):
    """Фаза A: backbone на естественном распределении + ClassBalancedFocalLoss."""
    print("\n=== Фаза A: обучение представления (естественное распределение) ===")
    stage1_state, best_lr, stage1_f1, _ = run_stage1(train_loader, val_loader, criterion)
    stage2_result = run_stage2(train_loader, val_loader, criterion, stage1_state, best_lr)
    print(
        f"\nФаза A завершена за {stage2_result['epochs_run']} эпох | "
        f"лучший val macro F1={stage2_result['best_f1']:.4f}"
    )
    return stage2_result, best_lr


def run_classifier_phase(representation_state, best_lr, train_loader, val_loader, criterion):
    """Фаза B: замороженный backbone, head на WeightedRandomSampler."""
    print("\n=== Фаза B: дообучение классификатора (backbone заморожен, balanced sampler) ===")
    torch.manual_seed(RANDOM_SEED)
    model = build_convnext_tiny(num_classes=len(CLASSES), freeze_backbone=True).to(device)
    model.load_state_dict(representation_state)
    freeze_backbone_only(model)

    classifier_lr = best_lr / CLASSIFIER_LR_DIVISOR
    print(
        f"LR={classifier_lr:.1e} | max_epochs={CLASSIFIER_MAX_EPOCHS} | "
        f"обучаемых параметров: {count_trainable_params(model):,}"
    )

    return train_loop(
        model, train_loader, val_loader, classifier_lr, criterion,
        max_epochs=CLASSIFIER_MAX_EPOCHS,
        min_epochs=CLASSIFIER_MIN_EPOCHS,
        patience=CLASSIFIER_PATIENCE,
        label="фазаB_head",
    )


def save_training_curves(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    epochs_range = range(1, len(history["val_loss"]) + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="train", alpha=0.8)
    axes[0].plot(epochs_range, history["val_loss"], label="val", alpha=0.8)
    axes[0].set_title("Loss (фаза B)")
    axes[0].set_xlabel("эпоха")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(epochs_range, history["val_f1"], label="macro F1", color="green", marker="o", markersize=3)
    axes[1].set_title("macro F1 (фаза B, val)")
    axes[1].set_xlabel("эпоха")
    axes[1].set_ylabel("macro F1")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, classes: list[str], path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("предсказание")
    ax.set_ylabel("истина")
    ax.set_title("Confusion matrix (test) — UBC building classifier")
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    set_random_seed()

    print("Устройство:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if not UBC_SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"{UBC_SPLIT_CSV} не найден. Сначала: python scripts/build_ubc_split.py"
        )

    split_df = load_split(UBC_SPLIT_CSV)
    train_df = split_df[split_df["split"] == "train"]
    class_counts = class_counts_from_df(train_df)

    print("Размер по split:")
    print(split_df["split"].value_counts())
    print("\nДисбаланс классов (весь набор / train):")
    print("весь набор:")
    print(split_df["class"].value_counts().to_string())
    print("train:")
    print(train_df["class"].value_counts().to_string())
    print(
        "\nНапоминание: industrial ~2% — свойство разметки UBC use_coarse "
        "(public/other не используются)."
    )

    eval_transform = get_transforms(image_size=IMAGE_SIZE)
    train_transform = get_building_train_transforms(image_size=IMAGE_SIZE, augment=True)

    train_ds = BuildingDataset(
        split_df, split="train", classes=CLASSES, transform=train_transform, use_mask=True
    )
    val_ds = BuildingDataset(
        split_df, split="val", classes=CLASSES, transform=eval_transform, use_mask=True
    )
    test_ds = BuildingDataset(
        split_df, split="test", classes=CLASSES, transform=eval_transform, use_mask=True
    )

    loader_workers = 4 if device.type == "cuda" else NUM_WORKERS
    loader_kwargs = dict(
        num_workers=loader_workers,
        persistent_workers=loader_workers > 0,
        pin_memory=device.type == "cuda",
    )

    natural_train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs
    )
    balanced_train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=make_weighted_sampler(train_df, CLASSES),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    print(f"\nОбучение: {len(train_ds)} | Валидация: {len(val_ds)} | Тест: {len(test_ds)}")
    print(
        f"Фаза A: ClassBalancedFocalLoss(beta={CLASS_BALANCED_BETA}, gamma={FOCAL_GAMMA}) | "
        f"естественное распределение"
    )
    print(
        f"Фаза B: FocalLoss(gamma={FOCAL_GAMMA}) | WeightedRandomSampler | "
        f"weight_decay={WEIGHT_DECAY}"
    )

    rep_criterion = make_class_balanced_criterion(class_counts)
    clf_criterion = make_focal_criterion()

    rep_result, best_lr = run_representation_phase(
        natural_train_loader, val_loader, rep_criterion
    )
    clf_result = run_classifier_phase(
        rep_result["best_state"],
        best_lr,
        balanced_train_loader,
        val_loader,
        clf_criterion,
    )

    print(f"\nФаза B завершена за {clf_result['epochs_run']} эпох")
    print(f"Лучший macro F1 на валидации (фаза B): {clf_result['best_f1']:.4f}")

    best_model_path = MODELS_DIR / CFG.model_filename
    torch.save(clf_result["best_state"], best_model_path)
    print(f"Модель сохранена: {best_model_path}")

    save_training_curves(clf_result["history"], REPORTS_DIR / CFG.training_curves_plot)

    final_model = build_convnext_tiny(num_classes=len(CLASSES), freeze_backbone=False).to(device)
    final_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    final_model.eval()

    val_preds, val_labels, _ = collect_predictions(val_loader, final_model)
    val_accuracy = float(np.mean(val_preds == val_labels))
    val_macro_f1 = float(f1_score(val_labels, val_preds, average="macro", zero_division=0))

    test_preds, test_labels, _ = collect_predictions(test_loader, final_model)
    test_accuracy = float(np.mean(test_preds == test_labels))
    test_macro_f1 = float(f1_score(test_labels, test_preds, average="macro", zero_division=0))

    print(f"\n=== Валидация === accuracy={val_accuracy:.4f} | macro F1={val_macro_f1:.4f}")
    print(f"=== Тест (исходный UBC val) === accuracy={test_accuracy:.4f} | macro F1={test_macro_f1:.4f}")
    print(classification_report(test_labels, test_preds, target_names=CLASSES, zero_division=0))

    save_confusion_matrix(test_labels, test_preds, CLASSES, REPORTS_DIR / CFG.confusion_matrix_plot)
    print(f"\nГрафики сохранены в {REPORTS_DIR}")


if __name__ == "__main__":
    main()

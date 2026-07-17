"""Функции потерь для обучения при дисбалансе классов.

FocalLoss           - focal loss + label smoothing, без учёта частоты классов
                       (годится, когда дисбаланс уже компенсирован сэмплером).
ClassBalancedFocalLoss - focal loss + label smoothing + per-class вес alpha,
                       посчитанный через "эффективное число примеров"
                       (Cui et al., 2019, "Class-Balanced Loss Based on
                       Effective Number of Samples"). Годится для обучения на
                       *естественном* (не ресэмплированном) распределении —
                       компенсирует дисбаланс через вес в loss, а не через
                       повторение редких примеров.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_balanced_alpha(
    class_counts: dict[str, int],
    classes: list[str],
    beta: float = 0.999,
) -> torch.Tensor:
    """Веса классов через эффективное число примеров: (1-beta)/(1-beta^n_c).

    Нормализуются так, чтобы среднее по классам было 1 (не завышает/занижает
    общий масштаб loss относительно label smoothing и learning rate).
    """
    counts = np.array([class_counts[c] for c in classes], dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    alpha = (1.0 - beta) / np.clip(effective_num, 1e-12, None)
    alpha = alpha / alpha.mean()
    return torch.tensor(alpha, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal loss с label smoothing, без per-class весов."""

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def _true_dist(self, log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = log_probs.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / (n_classes - 1) if n_classes > 1 else 0.0)
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        return true_dist

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        true_dist = self._true_dist(log_probs, targets)
        pt = (true_dist * log_probs.exp()).sum(dim=1).clamp(min=1e-8)
        focal = (1.0 - pt) ** self.gamma
        loss = -(focal * (true_dist * log_probs).sum(dim=1))
        return loss.mean()


class ClassBalancedFocalLoss(FocalLoss):
    """FocalLoss + per-class вес alpha (эффективное число примеров).

    Используется на этапе обучения представления (backbone) на естественном
    распределении: без него редкий класс почти не влияет на градиент
    (в среднем ~0.6 примера industrial на батч из 32 при дисбалансе 2%).
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__(gamma=gamma, label_smoothing=label_smoothing)
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        true_dist = self._true_dist(log_probs, targets)
        pt = (true_dist * log_probs.exp()).sum(dim=1).clamp(min=1e-8)
        focal = (1.0 - pt) ** self.gamma
        alpha_t = self.alpha.to(logits.device)[targets]
        loss = -(alpha_t * focal * (true_dist * log_probs).sum(dim=1))
        return loss.mean()

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multiclass focal loss with label smoothing and optional class weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            if class_weights.ndim != 1 or torch.any(class_weights <= 0):
                raise ValueError("class_weights must be a positive one-dimensional tensor")
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        classes = logits.shape[1]
        if classes < 2:
            raise ValueError("FocalLoss requires at least two classes")
        with torch.no_grad():
            smoothed = torch.full_like(logits, self.label_smoothing / (classes - 1))
            smoothed.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        log_probabilities = F.log_softmax(logits, dim=1)
        cross_entropy = -(smoothed * log_probabilities).sum(dim=1)
        probability = torch.exp(-cross_entropy)
        loss = ((1.0 - probability) ** self.gamma) * cross_entropy
        if self.class_weights is not None:
            if len(self.class_weights) != classes:
                raise ValueError("class_weights length must equal the number of classes")
            loss = loss * self.class_weights[targets]
        return loss.mean()

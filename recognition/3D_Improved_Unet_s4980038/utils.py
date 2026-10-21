"""Metrics, inference, profiling, and reproducibility helpers."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from modules import primary_logits


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SegmentationMetrics:
    """Accumulate a global confusion matrix and derive class metrics."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:

        # Compute the confusion matrix for the current batch and accumulate it
        prediction = prediction.detach().view(-1).cpu().long()
        target = target.detach().view(-1).cpu().long()
        valid = (target >= 0) & (target < self.num_classes)
        indices = self.num_classes * target[valid] + prediction[valid]
        matrix = torch.bincount(indices, minlength=self.num_classes**2)
        self.confusion += matrix.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, list[float] | float]:

        # Compute per-class Dice and IoU metrics from the accumulated confusion matrix
        true_positive = self.confusion.diag()
        target_count = self.confusion.sum(dim=1)
        prediction_count = self.confusion.sum(dim=0)
        dice_denominator = target_count + prediction_count
        union = target_count + prediction_count - true_positive
        dice = torch.where(
            dice_denominator > 0,
            2.0 * true_positive / dice_denominator,
            torch.ones_like(dice_denominator),
        )
        iou = torch.where(union > 0, true_positive / union, torch.ones_like(union))
        return {
            "dice_per_class": dice.tolist(),
            "iou_per_class": iou.tolist(),
            "mean_foreground_dice": dice[1:].mean().item(),
            "mean_foreground_iou": iou[1:].mean().item(),
        }

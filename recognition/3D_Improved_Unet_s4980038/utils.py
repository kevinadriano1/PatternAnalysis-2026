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


class BoundaryF1:
    """Tolerance-aware boundary F1 accumulator for a selected label.

    ``tolerance`` is a Chebyshev distance in voxels on the evaluated tensor
    grid, not millimetres.  With the default downsampled D,H,W shape, one voxel
    represents the resized grid spacing and may differ physically by case.
    """

    def __init__(self, label: int = 5, tolerance: int = 1) -> None:
        if not isinstance(tolerance, int) or tolerance < 0:
            raise ValueError("Boundary tolerance must be a non-negative integer")
        self.label = label
        self.tolerance = tolerance
        self.matched_prediction = 0.0
        self.prediction_boundary = 0.0
        self.matched_target = 0.0
        self.target_boundary = 0.0

    @staticmethod
    def _boundary(mask: torch.Tensor) -> torch.Tensor:

        # Erode the mask and subtract it from the original to get the boundary voxels
        mask = mask.float().unsqueeze(1)
        if mask.ndim == 5:
            eroded = 1.0 - F.max_pool3d(1.0 - mask, 3, stride=1, padding=1)
        elif mask.ndim == 4:
            eroded = 1.0 - F.max_pool2d(1.0 - mask, 3, stride=1, padding=1)
        else:
            raise ValueError(f"Expected batched 2D or 3D mask, got {mask.shape}")
        return (mask - eroded).clamp_min(0).bool()

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """Update the boundary F1 statistics with a new batch of predictions and targets."""

        pred_boundary = self._boundary(prediction == self.label)
        target_boundary = self._boundary(target == self.label)
        kernel = 2 * self.tolerance + 1
        if pred_boundary.ndim == 5:
            dilated_pred = F.max_pool3d(
                pred_boundary.float(), kernel, stride=1, padding=self.tolerance
            ).bool()
            dilated_target = F.max_pool3d(
                target_boundary.float(), kernel, stride=1, padding=self.tolerance
            ).bool()
        else:
            dilated_pred = F.max_pool2d(
                pred_boundary.float(), kernel, stride=1, padding=self.tolerance
            ).bool()
            dilated_target = F.max_pool2d(
                target_boundary.float(), kernel, stride=1, padding=self.tolerance
            ).bool()

        self.matched_prediction += (pred_boundary & dilated_target).sum().item()
        self.prediction_boundary += pred_boundary.sum().item()
        self.matched_target += (target_boundary & dilated_pred).sum().item()
        self.target_boundary += target_boundary.sum().item()

    def compute(self) -> dict[str, float]:
        """Compute the boundary precision, recall, and F1 score."""
        
        precision = self.matched_prediction / max(self.prediction_boundary, 1.0)
        recall = self.matched_target / max(self.target_boundary, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {
            "prostate_boundary_precision": precision,
            "prostate_boundary_recall": recall,
            "prostate_boundary_f1": f1,
        }
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

def dice_for_label(prediction: torch.Tensor, target: torch.Tensor, label: int) -> float:
    """Compute binary Dice for one label, returning 1 when both masks are empty."""

    prediction_mask = prediction == label
    target_mask = target == label
    denominator = prediction_mask.sum().item() + target_mask.sum().item()

    if denominator == 0:
        return 1.0
    intersection = (prediction_mask & target_mask).sum().item()
    return 2.0 * intersection / denominator


EXTENT_COUNT_KEYS = (
    "prostate_fp_voxels_below_extent",
    "prostate_fp_voxels_above_extent",
    "prostate_extra_slices_below_extent",
    "prostate_extra_slices_above_extent",
    "prostate_gt_positive_slices",
    "prostate_completely_missed_slices",
    "prostate_fp_voxels_on_gt_empty_slices",
)


def prostate_extent_metrics(
    prediction: torch.Tensor, target: torch.Tensor, label: int = 5
) -> dict[str, int | None]:
    """Count extent errors on the evaluation grid, using depth-index ordering.

    A complete miss means no prostate prediction anywhere on a GT-positive
    slice, not merely zero overlap. For an entirely GT-empty volume, the span
    is undefined; below/above counts are zero and all predictions are recorded
    by the GT-empty-slice FP count.
    """
    if prediction.ndim == 4 and prediction.shape[0] == 1:
        prediction, target = prediction[0], target[0]
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("Extent metrics require matching D,H,W tensors for one case")
    
    counts = (prediction == label).flatten(1).sum(1)
    gt_positive = (target == label).flatten(1).any(1)
    indices = torch.where(gt_positive)[0]

    # Initialize the result dictionary with default values and compute various extent metrics
    result = {key: 0 for key in EXTENT_COUNT_KEYS}
    result["prostate_gt_positive_slices"] = int(gt_positive.sum())
    result["prostate_completely_missed_slices"] = int((gt_positive & (counts == 0)).sum())
    result["prostate_fp_voxels_on_gt_empty_slices"] = int(counts[~gt_positive].sum())
    result.update(prostate_gt_first_slice=None, prostate_gt_last_slice=None)

    # If there are any GT-positive slices, record the first and last slice indices
    if indices.numel():
        first, last = int(indices[0]), int(indices[-1])
        result.update(prostate_gt_first_slice=first, prostate_gt_last_slice=last)
        for side, values in (("below", counts[:first]), ("above", counts[last + 1:])):
            result[f"prostate_fp_voxels_{side}_extent"] = int(values.sum())
            result[f"prostate_extra_slices_{side}_extent"] = int((values > 0).sum())
    return result


def regional_prostate_dice(
    prediction: torch.Tensor, target: torch.Tensor, label: int = 5
) -> dict[str, float | int | None]:
    """Measure prostate Dice in depth-low, centre, and depth-high regions.

    The first/last 20% of prostate-positive axial slices are reported as the
    two depth-end regions.  They must not be called anatomical apex/base
    without separately verifying the stored NIfTI orientation.
    """

    # Validate input shapes and ensure that the prediction and target tensors are 3D (D,H,W) for a single case
    if prediction.ndim == 4:
        if prediction.shape[0] != 1:
            raise ValueError("Regional metrics expect one case at a time")
        prediction = prediction[0]
        target = target[0]
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError("Regional metrics expect D,H,W prediction and target tensors")
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
        )
    positive_slices = torch.where((target == label).flatten(1).any(dim=1))[0]

    if positive_slices.numel() == 0:
        return {
            "prostate_end_low_dice": float("nan"),
            "prostate_centre_dice": float("nan"),
            "prostate_end_high_dice": float("nan"),
            "prostate_end_low_worst_slice": None,
            "prostate_centre_worst_slice": None,
            "prostate_end_high_worst_slice": None,
            "worst_prostate_slice": None,
            "worst_prostate_slice_dice": float("nan"),
        }

    # Determine the number of slices to include in the low and high depth regions (20% of positive slices, rounded)
    count = positive_slices.numel()
    end_count = max(1, round(count * 0.2))
    low = positive_slices[:end_count]
    high = positive_slices[-end_count:]
    centre = positive_slices[end_count:-end_count]

    if centre.numel() == 0:
        centre = positive_slices

    slice_scores = [
        (int(index), dice_for_label(prediction[index], target[index], label))
        for index in positive_slices
    ]
    worst_index, worst_score = min(slice_scores, key=lambda item: item[1])

    def worst_slice(indices: torch.Tensor) -> int:
        # Return the index of the slice with the worst Dice score among the given indices.
        return min(
            (int(index) for index in indices),
            key=lambda index: dice_for_label(prediction[index], target[index], label),
        )

    return {
        "prostate_end_low_dice": dice_for_label(prediction[low], target[low], label),
        "prostate_centre_dice": dice_for_label(prediction[centre], target[centre], label),
        "prostate_end_high_dice": dice_for_label(prediction[high], target[high], label),
        "prostate_end_low_worst_slice": worst_slice(low),
        "prostate_centre_worst_slice": worst_slice(centre),
        "prostate_end_high_worst_slice": worst_slice(high),
        "worst_prostate_slice": worst_index,
        "worst_prostate_slice_dice": worst_score,
    }
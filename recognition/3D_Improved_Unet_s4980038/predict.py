"""Evaluate a trained checkpoint, save NIfTI predictions, and visualise failures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import LABEL_NAMES, HipMRIVolumeDataset, discover_cases
from modules import build_model
from utils import (
    EXTENT_COUNT_KEYS,
    prostate_extent_metrics,
    BoundaryF1,
    SegmentationMetrics,
    dice_for_label,
    parameter_count,
    regional_prostate_dice,
    seed_everything,
    timed_inference,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-radius", type=int, default=0, help="Neighbouring evaluation slices on either side")
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--slice-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--failure-count", type=int, default=5)
    parser.add_argument(
        "--visualise-case",
        action="append",
        default=[],
        metavar="CASE_ID[:SLICE]",
        help=(
            "Generate matched figures for a test case, optionally at an exact "
            "downsampled depth index; may be supplied multiple times"
        ),
    )
    parser.add_argument(
        "--expected-model",
        choices=("unet2d", "improved_unet3d"),
        default=None,
        help="Fail if the checkpoint contains a different model type",
    )
    parser.add_argument(
        "--limit-cases", type=int, default=None, help="Development only: truncate the test split"
    )
    parser.add_argument("--no-save-nifti", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _parse_visualisation_specs(values: list[str]) -> list[tuple[str, int | None]]:
    """Parse --visualise-case values into (case_id, slice_index) tuples."""
    
    specs: list[tuple[str, int | None]] = []
    for value in values:
        case_id, separator, raw_slice = value.rpartition(":")
        if not separator:
            case_id, raw_slice = value, ""
        if not case_id:
            raise ValueError(f"Invalid --visualise-case value: {value!r}")
        try:
            slice_index = int(raw_slice) if raw_slice else None
        except ValueError as error:
            raise ValueError(
                f"Invalid slice in --visualise-case {value!r}; use CASE_ID[:SLICE]"
            ) from error
        specs.append((case_id, slice_index))
    return specs


def _case_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    accumulator = SegmentationMetrics(len(LABEL_NAMES))
    accumulator.update(prediction, target)
    values = accumulator.compute()
    dice = values["dice_per_class"]
    iou = values["iou_per_class"]
    if not isinstance(dice, list) or not isinstance(iou, list):
        raise TypeError("Expected per-class metric lists")
    result: dict[str, float] = {
        "mean_foreground_dice": float(values["mean_foreground_dice"]),
        "mean_foreground_iou": float(values["mean_foreground_iou"]),
    }
    for index, name in enumerate(LABEL_NAMES):
        result[f"dice_{name}"] = float(dice[index])
        result[f"iou_{name}"] = float(iou[index])
    return result


def _save_prediction(
    prediction: torch.Tensor,
    original_shape: tuple[int, int, int],
    affine: np.ndarray,
    output_path: Path,
) -> None:
    # Resize D,H,W back to Z,X,Y and transpose to NIfTI X,Y,Z.
    resized = F.interpolate(
        prediction[None, None].float(),
        size=(original_shape[2], original_shape[0], original_shape[1]),
        mode="nearest",
    )[0, 0]

    array = resized.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
    nib.save(nib.Nifti1Image(array, affine), str(output_path))


def _difference_rgb(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return an RGB image highlighting differences between prediction and target."""
    
    image = np.zeros((*target.shape, 3), dtype=np.float32)
    other_error = (prediction != target) & (prediction != 5) & (target != 5)
    false_positive = (prediction == 5) & (target != 5)
    false_negative = (prediction != 5) & (target == 5)

    image[other_error] = (1.0, 0.8, 0.0)
    image[false_positive] = (1.0, 0.1, 0.1)
    image[false_negative] = (0.1, 0.4, 1.0)

    return image


def save_failure_figure(candidate: dict[str, object], output_path: Path) -> None:
    
    # Validate candidate dictionary
    image = candidate["image"]
    target = candidate["target"]
    prediction = candidate["prediction"]
    slice_index = candidate["slice_index"]

    if not all(isinstance(value, torch.Tensor) for value in (image, target, prediction)):
        raise TypeError("Failure candidate arrays must be tensors")
    if not isinstance(slice_index, int):
        raise TypeError("Failure candidate slice index must be an integer")

    # Determine which slices to visualise based on the context radius
    radius = int(candidate.get("context_radius", 0))
    slices = range(max(0, slice_index - radius), min(target.shape[0], slice_index + radius + 1))
    colours = ["black", "#aaaaaa", "#f1c40f", "#3498db", "#e67e22", "#e84393"]
    cmap = ListedColormap(colours)
    figure, axes = plt.subplots(len(slices), 4, figsize=(14, 3.2 * len(slices)), squeeze=False)

    # Visualise each slice with the original image, ground truth, prediction, and error overlay
    for row, index in enumerate(slices):
        image_np, target_np, prediction_np = (x[index].numpy() for x in (image, target, prediction))
        for axis in axes[row]:
            axis.imshow(image_np, cmap="gray")
            axis.axis("off")
        axes[row, 0].set_title(f"MRI slice {index}" + (" (selected)" if index == slice_index else ""))
        
        for col, mask, title in ((1, target_np, "Ground truth"), (2, prediction_np, "Prediction")):
            axes[row, col].imshow(np.ma.masked_where(mask == 0, mask), cmap=cmap, vmin=0, vmax=5, alpha=0.55)
            axes[row, col].set_title(title)
        
        error = _difference_rgb(prediction_np, target_np)
        rgba = np.concatenate([error, (prediction_np != target_np)[..., None] * 0.7], axis=-1)
        axes[row, 3].imshow(rgba)
        axes[row, 3].set_title(f"Errors | prostate Dice {dice_for_label(prediction[index], target[index], label=5):.3f}")

    # Add a legend and title summarising the case and metrics
    handles = [Patch(color=colours[i], label=LABEL_NAMES[i]) for i in range(1, 6)]
    handles += [Patch(color=c, label=l) for c, l in (("red", "prostate FP"), ("#1a66ff", "prostate FN"), ("#ffcc00", "other label error"))]
    figure.legend(handles=handles, loc="lower center", ncol=4)
    figure.suptitle(f"{candidate['case_id']} | {candidate['region']} | GT span {candidate['extent']['prostate_gt_first_slice']}–{candidate['extent']['prostate_gt_last_slice']} | "
                   f"whole-volume prostate Dice {candidate['prostate_dice']:.3f}")
    
    figure.tight_layout(rect=(0, 0.07, 1, 0.96))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
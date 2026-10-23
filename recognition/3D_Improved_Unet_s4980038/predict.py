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
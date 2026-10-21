"""Leakage-free HipMRI NIfTI loading and preprocessing.

All weekly scans from a patient remain in exactly one split.  Volumes are
loaded lazily, robustly intensity-normalised, and downsampled with trilinear
image interpolation and nearest-neighbour label interpolation.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


LABEL_NAMES = ("background", "body", "bone", "bladder", "rectum", "prostate")
IMAGE_SUFFIX = "_LFOV.nii.gz"
LABEL_SUFFIX = "_SEMANTIC.nii.gz"


@dataclass(frozen=True)
class VolumeRecord:
    """Paths and identifiers for one aligned MRI/segmentation pair."""

    case_id: str
    patient_id: str
    image_path: str
    label_path: str

def discover_cases(data_root: str | Path) -> list[VolumeRecord]:
    """Find and validate all matching semantic MRI and label volumes."""
    root = Path(data_root).expanduser().resolve()
    image_dir = root / "semantic_MRs"
    label_dir = root / "semantic_labels_only"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Expected semantic_MRs and semantic_labels_only under {root}"
        )

    # Scan for all matching MRI and label files, and validate that each MRI has a label.
    records: list[VolumeRecord] = []
    missing: list[str] = []
    for image_path in sorted(image_dir.glob(f"*{IMAGE_SUFFIX}")):
        case_id = image_path.name[: -len(IMAGE_SUFFIX)]
        label_path = label_dir / f"{case_id}{LABEL_SUFFIX}"
        if not label_path.exists():
            missing.append(case_id)
            continue
        patient_id = case_id.split("_", maxsplit=1)[0]
        records.append(
            VolumeRecord(case_id, patient_id, str(image_path), str(label_path))
        )
    if missing:
        raise FileNotFoundError(f"Missing labels for {len(missing)} cases: {missing[:5]}")
    if not records:
        raise FileNotFoundError(f"No *{IMAGE_SUFFIX} files found in {image_dir}")
    return records


def split_cases(
    records: Sequence[VolumeRecord],
    seed: int = 3710,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, list[VolumeRecord]]:
    """Make a deterministic split over patients, never scans or slices."""

    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Training and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("A non-empty test fraction is required")

    patients = sorted({record.patient_id for record in records})
    if len(patients) < 3:
        raise ValueError("At least three patients are required for train/val/test splits")

    # Shuffle patients deterministically and split into train/validation/test sets
    random.Random(seed).shuffle(patients)
    train_end = max(1, round(len(patients) * train_fraction))
    val_count = max(1, round(len(patients) * validation_fraction))
    val_end = min(len(patients) - 1, train_end + val_count)
    train_patients = set(patients[:train_end])
    validation_patients = set(patients[train_end:val_end])
    test_patients = set(patients[val_end:])

    splits = {
        "train": [r for r in records if r.patient_id in train_patients],
        "validation": [r for r in records if r.patient_id in validation_patients],
        "test": [r for r in records if r.patient_id in test_patients],
    }

    patient_sets = [{r.patient_id for r in subset} for subset in splits.values()]
    if any(patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Patient leakage detected while constructing splits")
    return splits
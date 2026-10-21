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

def save_split_manifest(
    splits: dict[str, list[VolumeRecord]], output_path: str | Path
) -> None:
    """Persist the exact case-level split for reproducibility and audit."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        split: {
            "patients": sorted({record.patient_id for record in records}),
            "cases": [asdict(record) for record in records],
        }
        for split, records in splits.items()
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_manifest(path: str | Path) -> dict[str, list[VolumeRecord]]:
    """Load a manifest written by :func:`save_split_manifest`."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        split: [VolumeRecord(**item) for item in value["cases"]]
        for split, value in payload.items()
    }


def _load_nifti(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return its data and affine matrix."""
    
    image = nib.load(path)
    data = np.asarray(image.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got {data.shape} from {path}")
    return data, image.affine


def _left_right_flip_dims(affine: np.ndarray) -> tuple[int, int] | None:
    """Map the affine's left-right voxel axis to image and label tensor dims.

    NIfTI arrays are transposed from X,Y,Z to D,H,W = Z,X,Y.  The returned
    dimensions therefore include the image channel offset but not a batch
    dimension.  ``None`` disables flipping when the affine is ambiguous.
    """

    axis_codes = nib.aff2axcodes(affine)
    left_right_axes = [index for index, code in enumerate(axis_codes) if code in {"L", "R"}]
    if len(left_right_axes) != 1:
        return None
    nifti_axis = left_right_axes[0]
    label_dim = {0: 1, 1: 2, 2: 0}[nifti_axis]
    return label_dim + 1, label_dim


def audit_orientations(records: Sequence[VolumeRecord]) -> dict[str, object]:
    """Summarise NIfTI orientations and whether affine-driven LR flips are safe."""

    image_orientations: Counter[str] = Counter()
    label_orientations: Counter[str] = Counter()
    pair_mismatches: list[str] = []
    unsafe_flip_cases: list[str] = []

    # Scan all records and check for orientation mismatches and unsafe left-right flips.
    for record in records:
        image_affine = nib.load(record.image_path).affine
        label_affine = nib.load(record.label_path).affine
        image_codes = nib.aff2axcodes(image_affine)
        label_codes = nib.aff2axcodes(label_affine)

        image_orientations["".join(code or "?" for code in image_codes)] += 1
        label_orientations["".join(code or "?" for code in label_codes)] += 1

        if image_codes != label_codes:
            pair_mismatches.append(record.case_id)
        if _left_right_flip_dims(image_affine) is None:
            unsafe_flip_cases.append(record.case_id)

    return {
        "image_orientations": dict(sorted(image_orientations.items())),
        "label_orientations": dict(sorted(label_orientations.items())),
        "inconsistent_image_orientations": len(image_orientations) > 1,
        "image_label_orientation_mismatches": pair_mismatches,
        "left_right_flip_safe": not pair_mismatches and not unsafe_flip_cases,
        "left_right_flip_disabled_cases": unsafe_flip_cases,
    }


def _normalise_mri(image: torch.Tensor) -> torch.Tensor:
    """Clip and z-normalise the original finite, non-zero foreground only."""

    foreground_mask = image != 0
    valid_foreground_mask = foreground_mask & torch.isfinite(image)
    foreground = image[valid_foreground_mask]
    if foreground.numel() < 2:
        return torch.zeros_like(image)

    lower = torch.quantile(foreground, 0.005)
    upper = torch.quantile(foreground, 0.995)
    clipped_foreground = foreground.clamp(lower, upper)
    standard_deviation = clipped_foreground.std()

    normalised = torch.zeros_like(image)
    if not torch.isfinite(standard_deviation) or standard_deviation <= 1e-6:
        return normalised

    normalised[valid_foreground_mask] = (
        clipped_foreground - clipped_foreground.mean()
    ) / standard_deviation
    return normalised

def _validate_alignment(
    record: VolumeRecord,
    image: np.ndarray,
    label: np.ndarray,
    image_affine: np.ndarray,
    label_affine: np.ndarray,
) -> None:
    """Check that the image and label are compatible in shape, affine, and values."""

    if image.shape != label.shape:
        raise ValueError(
            f"Image/label shape mismatch for {record.case_id}: {image.shape} vs {label.shape}"
        )
    
    if not np.allclose(image_affine, label_affine, rtol=1e-4, atol=1e-4):
        raise ValueError(f"Image/label affine mismatch for {record.case_id}")
    unique = np.unique(label)

    if not np.allclose(unique, np.round(unique)):
        raise ValueError(f"Non-integer segmentation values {unique.tolist()} in {record.label_path}")
    
    if unique.min(initial=0) < 0 or unique.max(initial=0) >= len(LABEL_NAMES):
        raise ValueError(f"Unexpected labels {unique.tolist()} in {record.label_path}")
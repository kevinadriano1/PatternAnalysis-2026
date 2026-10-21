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
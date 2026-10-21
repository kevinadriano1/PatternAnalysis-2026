"""Train, validate, test, and save either HipMRI segmentation model."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import (
    LABEL_NAMES,
    HipMRIVolumeDataset,
    audit_orientations,
    discover_cases,
    parse_spatial_size,
    save_split_manifest,
    split_cases,
)
from modules import DiceCrossEntropyLoss, build_model, primary_logits, supervised_loss
from utils import BoundaryF1, SegmentationMetrics, infer_volume, parameter_count, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    # parse command line arguments

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("unet2d", "improved_unet3d"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 128, 128), metavar=("D", "H", "W"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1, help="Volumes per batch")
    parser.add_argument("--slice-batch-size", type=int, default=16, help="2D slices per forward pass")
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="Development only: truncate each split after patient-level splitting",
    )
    return parser.parse_args()


def _move_batch(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Move a batch of data to the specified device and ensure that the image and label are tensors.
    
    image = batch["image"]
    target = batch["label"]
    if not isinstance(image, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("Data loader must return tensor image and label fields")
    return image.to(device, non_blocking=True), target.to(device, non_blocking=True)


def _two_dimensional_step(
    model: torch.nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    criterion: torch.nn.Module,
    slice_batch_size: int,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, float, torch.Tensor]:
    """Process one volume batch as independent axial slices."""

    batch, channels, depth, height, width = image.shape
    slices = image.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, height, width)
    labels = target.reshape(batch * depth, height, width)
    predictions: list[torch.Tensor] = []
    weighted_loss = 0.0
    total_slices = slices.shape[0]

    # loop over slices in batches to avoid memory issues
    for start in range(0, total_slices, slice_batch_size):
        end = min(start + slice_batch_size, total_slices)
        fraction = (end - start) / total_slices
        with torch.autocast(
            device_type=image.device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(slices[start:end])
            loss = criterion(primary_logits(logits), labels[start:end])
        if scaler is not None:
            scaler.scale(loss * fraction).backward()
        weighted_loss += loss.detach().item() * fraction
        predictions.append(primary_logits(logits).detach().argmax(dim=1))

    prediction = torch.cat(predictions).reshape(batch, depth, height, width)
    return weighted_loss, weighted_loss, prediction
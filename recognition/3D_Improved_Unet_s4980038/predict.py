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



def main() -> None:

    # Parse command line arguments and validate them
    args = parse_args()
    if args.failure_count < 1 or args.failure_count > 5:
        raise ValueError("--failure-count must be between 1 and 5")
    if args.context_radius < 0:
        raise ValueError("--context-radius must be non-negative")

    # Parse visualisation specifications and prepare output directories
    visualisation_specs = _parse_visualisation_specs(args.visualise_case)
    requested_case_ids = {case_id for case_id, _ in visualisation_specs}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "nifti_predictions"
    figure_dir = args.output_dir / "failure_cases"
    
    if not args.no_save_nifti:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_name = str(checkpoint["model_name"])
    
    if args.expected_model is not None and model_name != args.expected_model:
        raise ValueError(
            f"Expected a {args.expected_model} checkpoint, but {args.checkpoint} "
            f"contains {model_name}"
        )
    seed = int(checkpoint.get("seed", 3710))
    spatial_size = tuple(int(value) for value in checkpoint["spatial_size"])

    # Set random seeds for reproducibility
    seed_everything(seed)

    # Discover test cases and validate the split manifest
    records = discover_cases(args.data_root)
    manifest_path = args.split_manifest or args.checkpoint.parent / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patient_sets = [{row["patient_id"] for row in manifest[key]["cases"]} for key in ("train", "validation", "test")]

    # Check for patient leakage across splits
    if any(patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Patient leakage in saved split manifest")
    lookup = {record.case_id: record for record in records}
    saved_test = manifest["test"]["cases"]
    
    if not saved_test or len({r["case_id"] for r in saved_test}) != len(saved_test):
        raise ValueError("Empty or duplicated test cases in manifest")
    test_records = []

    # Validate that the patient IDs in the manifest match the discovered records
    for row in saved_test:
        record = lookup[row["case_id"]]
        if record.patient_id != row["patient_id"]:
            raise ValueError(f"Patient mapping changed for {record.case_id}")
        test_records.append(record)
    if args.limit_cases is not None:
        test_records = test_records[: args.limit_cases]

    # Prepare the dataset and data loader for evaluation
    dataset = HipMRIVolumeDataset(test_records, spatial_size=spatial_size, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = build_model(
        model_name,
        num_classes=int(checkpoint["num_classes"]),
        base_channels=checkpoint.get("base_channels"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    use_amp = device.type == "cuda" and not args.no_amp

    # Initialize metrics accumulators and prepare for evaluation
    aggregate = SegmentationMetrics(len(LABEL_NAMES))
    boundary = BoundaryF1(label=5, tolerance=1)
    rows: list[dict[str, object]] = []
    lowest_case_by_patient: dict[str, dict[str, object]] = {}
    requested_candidates: dict[str, dict[str, object]] = {}
    latencies: list[float] = []
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Evaluate the model on the test dataset and collect metrics
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            prediction, latency_ms = timed_inference(
                model, image, model_name, args.slice_batch_size, use_amp
            )
            latencies.append(latency_ms)
            aggregate.update(prediction, target)
            boundary.update(prediction, target)

            case_id = str(batch["case_id"][0])
            patient_id = str(batch["patient_id"][0])
            metrics = _case_metrics(prediction, target)
            regional = regional_prostate_dice(prediction, target, label=5)
            extent = prostate_extent_metrics(prediction, target)
            row: dict[str, object] = {
                "case_id": case_id,
                "patient_id": patient_id,
                "latency_ms": latency_ms,
                **metrics,
                **regional,
                **extent,
            }
            rows.append(row)

            original_shape = tuple(int(value) for value in batch["original_shape"][0].tolist())
            affine = batch["affine"][0].numpy()
            if not args.no_save_nifti:
                _save_prediction(
                    prediction[0],
                    original_shape,
                    affine,
                    prediction_dir / f"{case_id}_prediction.nii.gz",
                )
            # Store the candidate information for potential failure visualisation
            candidate = {
                "case_id": case_id,
                "patient_id": patient_id,
                "prostate_dice": metrics["dice_prostate"],
                "image": image[0, 0].detach().cpu(),
                "target": target[0].detach().cpu(),
                "prediction": prediction[0].detach().cpu(),
                "regional": regional,
                "extent": extent,
                "axis_codes": "".join(nib.aff2axcodes(affine)),
                "context_radius": args.context_radius,
            }
            existing = lowest_case_by_patient.get(patient_id)
            if existing is None or float(candidate["prostate_dice"]) < float(
                existing["prostate_dice"]
            ):
                lowest_case_by_patient[patient_id] = candidate
            if case_id in requested_case_ids:
                requested_candidates[case_id] = candidate

    # Write per-case metrics to CSV if any rows were collected
    if rows:
        with (args.output_dir / "per_case_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    missing_requested = sorted(requested_case_ids - requested_candidates.keys())
    if missing_requested:
        raise ValueError(
            "Requested visualisation cases are not in the evaluated test split: "
            + ", ".join(missing_requested)
        )

    if visualisation_specs:
        selected_specs = [
            (requested_candidates[case_id], slice_index)
            for case_id, slice_index in visualisation_specs
        ]
    else:
        ranked_candidates = sorted(
            lowest_case_by_patient.values(),
            key=lambda item: float(item["prostate_dice"]),
        )[: args.failure_count]
        selected_specs = [(candidate, None) for candidate in ranked_candidates]

    figure_records: list[dict[str, object]] = []
    selected_failure_volumes: list[dict[str, object]] = []
    selected_volume_ids: set[str] = set()
    
    figure_rank = 0

    # Generate failure figures and collect records for the summary
    for candidate, requested_slice in selected_specs:
        candidate_case_id = str(candidate["case_id"])
        if candidate_case_id not in selected_volume_ids:
            selected_volume_ids.add(candidate_case_id)
            selected_failure_volumes.append(
                {
                    "case_id": candidate_case_id,
                    "patient_id": candidate["patient_id"],
                    "whole_volume_prostate_dice": candidate["prostate_dice"],
                }
            )
        regional = candidate["regional"]
        if not isinstance(regional, dict):
            raise TypeError("Candidate regional metrics must be a dictionary")
        if requested_slice is not None:
            region_slices = [("requested", requested_slice)]
        else:
            region_slices = [
                ("depth-low", regional["prostate_end_low_worst_slice"]),
                ("central", regional["prostate_centre_worst_slice"]),
                ("depth-high", regional["prostate_end_high_worst_slice"]),
            ]
            if all(slice_index is None for _, slice_index in region_slices):
                region_slices = [("central fallback", spatial_size[0] // 2)]

        # Track seen slices to avoid duplicate visualisations for the same slice
        seen_slices: set[int] = set()
        for region, raw_slice in region_slices:
            if raw_slice is None:
                continue
            slice_index = int(raw_slice)
            if slice_index < 0 or slice_index >= spatial_size[0]:
                raise ValueError(
                    f"Slice {slice_index} for {candidate['case_id']} is outside "
                    f"the valid range 0..{spatial_size[0] - 1}"
                )
            if requested_slice is None and slice_index in seen_slices:
                continue
            seen_slices.add(slice_index)
            target = candidate["target"]
            prediction = candidate["prediction"]
            if not isinstance(target, torch.Tensor) or not isinstance(
                prediction, torch.Tensor
            ):
                raise TypeError("Candidate prediction and target must be tensors")
            figure_candidate = {
                **candidate,
                "region": region,
                "slice_index": slice_index,
                "slice_dice": dice_for_label(
                    prediction[slice_index], target[slice_index], label=5
                ),
            }
            figure_rank += 1
            safe_region = region.replace(" ", "_")
            save_failure_figure(
                figure_candidate,
                figure_dir
                / (
                    f"{figure_rank:02d}_{candidate['case_id']}_{safe_region}_"
                    f"slice_{slice_index}.png"
                ),
            )
            # Record the figure information for the summary JSON
            figure_records.append(
                {
                    "case_id": candidate["case_id"],
                    "patient_id": candidate["patient_id"],
                    "region": region,
                    "gt_first_slice": candidate["extent"]["prostate_gt_first_slice"],
                    "gt_last_slice": candidate["extent"]["prostate_gt_last_slice"],
                    "nifti_axis_codes": candidate["axis_codes"],
                    "context_radius": args.context_radius,
                    "slice_index": slice_index,
                    "whole_volume_prostate_dice": candidate["prostate_dice"],
                    "slice_prostate_dice": figure_candidate["slice_dice"],
                }
            )

    # Write a summary JSON file with aggregated metrics and evaluation details
    summary: dict[str, object] = {
        "extent_totals": {key: sum(int(row[key]) for row in rows) for key in EXTENT_COUNT_KEYS},
        "extent_definition": "Downsampled grid counts; below/above mean depth indices. Complete miss means no prostate prediction on a GT-positive slice.",
        "spatial_size": list(spatial_size),
        "use_amp": use_amp,
        "torch_version": str(torch.__version__),
        "split_manifest": str(manifest_path),
        "model": model_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "test_patients": sorted({record.patient_id for record in test_records}),
        "test_case_ids": [record.case_id for record in test_records],
        "test_cases": len(test_records),
        **aggregate.compute(),
        **boundary.compute(),
        "parameters": parameter_count(model),
        "mean_inference_ms_per_case": float(np.mean(latencies)) if latencies else float("nan"),
        "peak_inference_gpu_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "peak_gpu_vram_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        ),
        "peak_training_gpu_allocated_mib": checkpoint.get(
            "peak_training_gpu_allocated_mib"
        ),
        "device": str(device),
        "gpu_model": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "boundary_tolerance_voxels": 1,
        "boundary_tolerance_definition": (
            "Chebyshev distance on the downsampled D,H,W evaluation grid; not millimetres"
        ),
        "failure_cases": selected_failure_volumes,
        "failure_figures": figure_records,
    }
    write_json(summary, args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
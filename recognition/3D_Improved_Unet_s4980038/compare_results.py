"""Validate and compare baseline and proposed-model prediction results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from dataset import LABEL_NAMES
from utils import EXTENT_COUNT_KEYS


def _read_case_rows(path: Path) -> dict[str, dict[str, str]]:
    """Read per-case evaluation CSV and return a mapping from case_id to row dict."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"case_id", "patient_id"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing columns in {path}: {sorted(missing_columns)}")
        rows = list(reader)
    
    if not rows:
        raise ValueError(f"No evaluated cases found in {path}")
    case_ids = [row.get("case_id", "") for row in rows]
    duplicates = sorted(case_id for case_id, count in Counter(case_ids).items() if count > 1)
    
    if "" in case_ids:
        raise ValueError(f"Missing case_id value in {path}")
    if duplicates:
        raise ValueError(f"Duplicated case_id values in {path}: {duplicates}")
    return {row["case_id"]: row for row in rows}


def _validate_evaluation_sets(
    baseline_summary: dict[str, object],
    proposed_summary: dict[str, object],
    baseline_cases: dict[str, dict[str, str]],
    proposed_cases: dict[str, dict[str, str]],
) -> list[str]:
    """Validate that the baseline and proposed evaluation sets are consistent and return sorted case IDs."""
    
    baseline_ids = set(baseline_cases)
    proposed_ids = set(proposed_cases)
    missing_from_proposed = sorted(baseline_ids - proposed_ids)
    additional_in_proposed = sorted(proposed_ids - baseline_ids)
    
    if missing_from_proposed or additional_in_proposed:
        raise ValueError(
            "Evaluation case mismatch. "
            f"Missing from proposed: {missing_from_proposed}; "
            f"additional in proposed: {additional_in_proposed}"
        )

    baseline_patients = {row["patient_id"] for row in baseline_cases.values()}
    proposed_patients = {row["patient_id"] for row in proposed_cases.values()}
    
    if baseline_patients != proposed_patients:
        raise ValueError(
            "Evaluation patient mismatch. "
            f"Baseline-only: {sorted(baseline_patients - proposed_patients)}; "
            f"proposed-only: {sorted(proposed_patients - baseline_patients)}"
        )
    for case_id in baseline_ids:
        if baseline_cases[case_id]["patient_id"] != proposed_cases[case_id]["patient_id"]:
            raise ValueError(f"Patient ID differs between result files for {case_id}")

    # Validate that the summary JSON files match the per-case CSVs
    for name, summary, case_ids, patients in (
        ("baseline", baseline_summary, baseline_ids, baseline_patients),
        ("proposed", proposed_summary, proposed_ids, proposed_patients),
    ):
        if int(summary["test_cases"]) != len(case_ids):
            raise ValueError(
                f"{name} summary says {summary['test_cases']} test cases but its CSV has "
                f"{len(case_ids)}"
            )
        summary_patients = set(summary.get("test_patients", []))
        if summary_patients != patients:
            raise ValueError(
                f"{name} summary patient set does not match its per-case CSV: "
                f"summary={sorted(summary_patients)}, CSV={sorted(patients)}"
            )
        summary_case_ids = set(summary.get("test_case_ids", case_ids))
        if summary_case_ids != case_ids:
            raise ValueError(f"{name} summary case IDs do not match its per-case CSV")
    return sorted(baseline_ids)


def _summary_row(summary: dict[str, object]) -> dict[str, object]:
    """Convert a summary JSON dict to a single-row dict for CSV output."""

    dice = summary["dice_per_class"]
    iou = summary["iou_per_class"]
    if not isinstance(dice, list) or not isinstance(iou, list):
        raise TypeError("Summary per-class metrics must be lists")
    
    # Construct a row dictionary with relevant metrics and information
    row: dict[str, object] = {
        "model": summary["model"],
        "mean_foreground_dice": summary["mean_foreground_dice"],
        "mean_foreground_iou": summary["mean_foreground_iou"],
        "prostate_dice": dice[5],
        "prostate_iou": iou[5],
        "prostate_boundary_f1": summary["prostate_boundary_f1"],
        "parameters": summary["parameters"],
        "peak_training_gpu_allocated_mib": summary.get(
            "peak_training_gpu_allocated_mib"
        ),
        "peak_inference_gpu_allocated_mib": summary.get(
            "peak_inference_gpu_allocated_mib", summary.get("peak_gpu_vram_mb")
        ),
        "mean_inference_ms_per_case": summary["mean_inference_ms_per_case"],
        "test_cases": summary["test_cases"],
        "comparison_scope_note": (
            "Models differ in dimensionality and architecture; the proposed model also uses "
            "residual units and deep supervision."
        ),
    }
    row.update(summary.get("extent_totals", {}))
    
    for index, name in enumerate(LABEL_NAMES):
        row[f"dice_{name}"] = dice[index]
        row[f"iou_{name}"] = iou[index]
    return row
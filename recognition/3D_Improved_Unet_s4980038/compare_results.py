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



def _difference_row(baseline: dict[str, object], proposed: dict[str, object]) -> dict[str, object]:
    """Compute the difference between proposed and baseline summary rows for each metric."""
    
    row: dict[str, object] = {"model": "proposed_minus_baseline"}
    
    for key in baseline:
        if key == "model":
            continue
        if key == "comparison_scope_note":
            row[key] = baseline[key]
            continue
        baseline_value = baseline[key]
        proposed_value = proposed[key]
        
        if isinstance(baseline_value, (int, float)) and isinstance(
            proposed_value, (int, float)
        ):
            row[key] = proposed_value - baseline_value
        else:
            row[key] = None
    return row


def _write_case_comparison(
    case_ids: list[str],
    baseline_cases: dict[str, dict[str, str]],
    proposed_cases: dict[str, dict[str, str]],
    output: Path,
) -> list[dict[str, object]]:
    """Write a CSV comparing baseline and proposed per-case metrics, returning the rows."""
    
    metric_names = (
        "dice_prostate",
        "iou_prostate",
        "prostate_end_low_dice",
        "prostate_centre_dice",
        "prostate_end_high_dice",
        "latency_ms",
    )
    
    available = [key for key in EXTENT_COUNT_KEYS if all(key in row for row in [*baseline_cases.values(), *proposed_cases.values()])]
    metric_names += tuple(available)
    rows: list[dict[str, object]] = []

    # Iterate through each case ID and compute the differences between baseline and proposed metrics
    for case_id in case_ids:
        baseline = baseline_cases[case_id]
        proposed = proposed_cases[case_id]
        row: dict[str, object] = {
            "case_id": case_id,
            "patient_id": baseline["patient_id"],
        }
        for metric in metric_names:
            baseline_value = float(baseline[metric])
            proposed_value = float(proposed[metric])
            row[f"baseline_{metric}"] = baseline_value
            row[f"proposed_{metric}"] = proposed_value
            row[f"delta_{metric}"] = proposed_value - baseline_value
        rows.append(row)

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:

    # Parse command-line arguments for baseline and proposed evaluation results, and output path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--proposed", type=Path, required=True)
    parser.add_argument("--baseline-cases", type=Path, default=None)
    parser.add_argument("--proposed-cases", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    
    args = parser.parse_args()

    baseline_summary = json.loads(args.baseline.read_text(encoding="utf-8"))
    proposed_summary = json.loads(args.proposed.read_text(encoding="utf-8"))
    baseline_case_path = args.baseline_cases or args.baseline.with_name(
        "per_case_metrics.csv"
    )
    proposed_case_path = args.proposed_cases or args.proposed.with_name(
        "per_case_metrics.csv"
    )

    # Read per-case evaluation CSVs for baseline and proposed models
    baseline_cases = _read_case_rows(baseline_case_path)
    proposed_cases = _read_case_rows(proposed_case_path)
    case_ids = _validate_evaluation_sets(
        baseline_summary, proposed_summary, baseline_cases, proposed_cases
    )

    if set(baseline_summary.get("extent_totals", {})) != set(proposed_summary.get("extent_totals", {})):
        raise ValueError("Extent metrics missing from one model; rerun both evaluations")
    for key in ("spatial_size", "use_amp"):
        if baseline_summary.get(key) != proposed_summary.get(key):
            raise ValueError(f"Evaluation settings differ: {key}")
    
    baseline_row = _summary_row(baseline_summary)
    proposed_row = _summary_row(proposed_summary)
    comparison_rows = [
        baseline_row,
        proposed_row,
        _difference_row(baseline_row, proposed_row),
    ]

    # Write the summary comparison CSV and per-case comparison CSV, then generate a report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(baseline_row))
        writer.writeheader()
        writer.writerows(comparison_rows)

    case_output = args.output.with_name(
        f"{args.output.stem}_per_case{args.output.suffix}"
    )
    case_rows = _write_case_comparison(
        case_ids, baseline_cases, proposed_cases, case_output
    )
    ranked = sorted(case_rows, key=lambda row: float(row["delta_dice_prostate"]))

    # Generate a JSON report summarizing the evaluation results and differences between baseline and proposed models
    report = {
        "validated_case_count": len(case_ids),
        "validated_patient_count": len({row["patient_id"] for row in case_rows}),
        "summary_rows": comparison_rows,
        "largest_3d_declines": ranked[:3],
        "largest_3d_improvements": list(reversed(ranked[-3:])),
        "case_comparison_csv": str(case_output),
        "interpretation_note": (
            "Observed differences are associated with the complete model configurations, not "
            "dimensionality alone: the proposed model also contains residual units and deep "
            "supervision."
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

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


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    model_name: str,
    slice_batch_size: int,
    amp_enabled: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, object]:
    """Run one epoch of training or evaluation."""
    
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(len(LABEL_NAMES))
    boundary = BoundaryF1(label=5, tolerance=1)
    total_main_loss = 0.0
    total_optimization_loss = 0.0
    total_cases = 0

    grad_context = torch.enable_grad() if training else torch.inference_mode()

    with grad_context:
        for batch in loader:
            image, target = _move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            # Use different processing for 2D and 3D models
            if model_name == "unet2d":
                batch_main_loss, batch_optimization_loss, prediction = _two_dimensional_step(
                    model,
                    image,
                    target,
                    criterion,
                    slice_batch_size,
                    amp_enabled,
                    scaler if training else None,
                )

            else:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    output = model(image)
                    optimization_loss = supervised_loss(output, target, criterion)
                    if training:
                        # Reporting-only loss: detached so the optimisation
                        # graph and deep-supervision objective are unchanged.
                        main_loss = criterion(primary_logits(output).detach(), target)
                    else:
                        main_loss = optimization_loss

                if training and scaler is not None:
                    scaler.scale(optimization_loss).backward()

                
                batch_main_loss = main_loss.detach().item()
                batch_optimization_loss = optimization_loss.detach().item()
                prediction = primary_logits(output).detach().argmax(dim=1)

            if training:
                if scaler is None:
                    raise RuntimeError("A gradient scaler is required during training")
                scaler.step(optimizer)
                scaler.update()

            # Update metrics and accumulate losses
            metrics.update(prediction, target)
            if not training:
                boundary.update(prediction, target)
            cases = image.shape[0]
            total_main_loss += batch_main_loss * cases
            total_optimization_loss += batch_optimization_loss * cases
            total_cases += cases

    # Compute average losses and return results
    main_loss = total_main_loss / max(total_cases, 1)
    optimization_loss = total_optimization_loss / max(total_cases, 1)
    result: dict[str, object] = {
        "main_loss": main_loss,
        "optimization_loss": optimization_loss,
        "loss": main_loss,
        **metrics.compute(),
    }
    if not training:
        result.update(boundary.compute())
    return result


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    """Save training history to CSV and plot training curves."""

    fieldnames = list(history[0])
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_main_loss"] for row in history], label="train main")
    axes[0].plot(
        epochs,
        [row["validation_main_loss"] for row in history],
        label="validation main",
    )
    if any(
        abs(row["train_optimization_loss"] - row["train_main_loss"]) > 1e-8
        for row in history
    ):
        axes[0].plot(
            epochs,
            [row["train_optimization_loss"] for row in history],
            linestyle="--",
            label="train total (with auxiliary)",
        )

    axes[0].set(title="Segmentation and optimisation loss", xlabel="Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_foreground_dice"] for row in history], label="train")
    axes[1].plot(epochs, [row["validation_foreground_dice"] for row in history], label="validation")
    axes[1].set(title="Mean foreground Dice", xlabel="Epoch", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(figure)


def profile_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    slice_batch_size: int,
    amp_enabled: bool,
) -> dict[str, float | int | str]:
    """Profile model inference performance."""

    model.eval()
    timings: list[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            image, _ = _move_batch(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            _ = infer_volume(model, image, model_name, slice_batch_size, amp_enabled)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - start) * 1000.0 / image.shape[0]
            if batch_index > 0:  # discard first case as a kernel warm-up
                timings.append(elapsed)
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )

    return {
        "parameters": parameter_count(model),
        "mean_inference_ms_per_case": sum(timings) / max(len(timings), 1),
        "peak_inference_gpu_allocated_mib": peak_mb,
        "peak_gpu_vram_mb": peak_mb,
        "device": str(device),
        "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }

def main() -> None:
    """Main training loop for HipMRI segmentation."""

    # Parse command line arguments and set up training environment
    args = parse_args()
    spatial_size = parse_spatial_size(args.spatial_size)
    if args.epochs < 1 or args.batch_size < 1 or args.slice_batch_size < 1:
        raise ValueError("Epoch and batch sizes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp

    records = discover_cases(args.data_root)
    orientation_audit = audit_orientations(records)
    splits = split_cases(records, seed=args.seed)
    save_split_manifest(splits, args.output_dir / "split_manifest.json")
    if args.limit_cases is not None:
        splits = {name: subset[: args.limit_cases] for name, subset in splits.items()}

    # Create datasets and data loaders for training, validation, and testing
    datasets = {
        name: HipMRIVolumeDataset(
            subset,
            spatial_size=spatial_size,
            augment=name == "train",
        )
        for name, subset in splits.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        for name, dataset in datasets.items()
    }

    # Build the model, define the loss function, optimizer, and learning rate scheduler
    model = build_model(args.model, len(LABEL_NAMES), args.base_channels).to(device)
    # Rare OARs and prostate receive more influence than background/body.
    weights = torch.tensor([0.05, 0.35, 1.0, 2.0, 2.0, 4.0], device=device)
    criterion = DiceCrossEntropyLoss(weights, dice_weight=0.6)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    run_config = {
        **vars(args),
        "data_root": str(args.data_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "spatial_size": list(spatial_size),
        "device": str(device),
        "amp": amp_enabled,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "orientation_audit": orientation_audit,
        "split_case_counts": {name: len(items) for name, items in splits.items()},
        "split_patient_counts": {
            name: len({item.patient_id for item in items}) for name, items in splits.items()
        },
    }
    write_json(run_config, args.output_dir / "run_config.json")
    print(json.dumps(run_config, indent=2), flush=True)

    best_score = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    checkpoint_path = args.output_dir / "best_model.pt"
    peak_training_gpu_allocated_mib = 0.0

    # Main training loop over epochs
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            args.model,
            args.slice_batch_size,
            amp_enabled,
            optimizer,
            scaler,
        )
        epoch_training_peak_mib = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        peak_training_gpu_allocated_mib = max(
            peak_training_gpu_allocated_mib, epoch_training_peak_mib
        )
        validation_metrics = run_epoch(
            model,
            loaders["validation"],
            criterion,
            device,
            args.model,
            args.slice_batch_size,
            amp_enabled,
        )
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "train_main_loss": float(train_metrics["main_loss"]),
            "validation_main_loss": float(validation_metrics["main_loss"]),
            "train_optimization_loss": float(train_metrics["optimization_loss"]),
            "validation_optimization_loss": float(
                validation_metrics["optimization_loss"]
            ),
            "train_foreground_dice": float(train_metrics["mean_foreground_dice"]),
            "validation_foreground_dice": float(validation_metrics["mean_foreground_dice"]),
            "training_peak_gpu_allocated_mib": epoch_training_peak_mib,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        score = row["validation_foreground_dice"]
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": args.model,
                    "model_state": model.state_dict(),
                    "num_classes": len(LABEL_NAMES),
                    "base_channels": args.base_channels,
                    "spatial_size": list(spatial_size),
                    "seed": args.seed,
                    "label_names": list(LABEL_NAMES),
                    "best_validation_foreground_dice": best_score,
                    "peak_training_gpu_allocated_mib": peak_training_gpu_allocated_mib,
                    "gpu_model": (
                        torch.cuda.get_device_name(device) if device.type == "cuda" else None
                    ),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs", flush=True)
            break

    # Save training history, best model checkpoint, and test metrics
    save_history(history, args.output_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint["peak_training_gpu_allocated_mib"] = peak_training_gpu_allocated_mib
    checkpoint["gpu_model"] = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else None
    )
    
    torch.save(checkpoint, checkpoint_path)
    training_summary = {
        "model": args.model,
        "epochs_completed": len(history),
        "best_validation_foreground_dice": best_score,
        "peak_training_gpu_allocated_mib": peak_training_gpu_allocated_mib,
        "gpu_model": checkpoint["gpu_model"],
        "device": str(device),
        "memory_definition": "Peak PyTorch GPU memory allocated during a training epoch",
    }
    write_json(training_summary, args.output_dir / "training_summary.json")
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        args.model,
        args.slice_batch_size,
        amp_enabled,
    )

    test_metrics["labels"] = list(LABEL_NAMES)
    test_metrics["resources"] = profile_inference(
        model,
        loaders["test"],
        device,
        args.model,
        args.slice_batch_size,
        amp_enabled,
    )

    test_metrics["resources"][
        "peak_training_gpu_allocated_mib"
    ] = peak_training_gpu_allocated_mib
    write_json(test_metrics, args.output_dir / "test_metrics.json")
    print(json.dumps(test_metrics, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted", file=sys.stderr)
        raise SystemExit(130)

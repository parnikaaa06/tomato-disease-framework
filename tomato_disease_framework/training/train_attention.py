"""Training and validation loops for EfficientNetB0 with dynamic attention."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from evaluation.classification_metrics import classification_metrics
from models.efficientnet_dynamic_attention import EfficientNetB0DynamicAttention
from preprocessing.dataset import TomatoLeafDataset, read_split_csv
from preprocessing.transforms import build_transforms


def set_reproducibility(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(config: dict[str, Any]) -> torch.device:
    """Resolve the training device from configuration."""
    requested_device = config["runtime"].get("device", "auto")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if requested_device == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    """Calculate validation loss and classification metrics."""
    model.eval()
    total_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        all_predictions.extend(predictions.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    metrics = classification_metrics(
        all_targets,
        all_predictions,
        class_names,
    )
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
) -> None:
    """Save the selected model and optimizer state."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        output,
    )


def save_experiment_config(output_dir: str | Path, config: dict[str, Any]) -> None:
    """Save the configuration used for the experiment."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def save_training_history(
    output_dir: str | Path,
    history: list[dict[str, Any]],
) -> None:
    """Persist accumulated epoch metrics so progress survives interruption."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/phase3_attention.yaml",
        help="Path to the Phase 3 attention configuration YAML.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the epoch count in the experiment configuration.",
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    experiment = config["experiment"]
    if args.epochs is not None:
        experiment["epochs"] = args.epochs
    data_config = config["data"]
    runtime = config["runtime"]

    set_reproducibility(experiment["seed"])
    device = resolve_device(config)

    print("Configuration loaded.")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_rows = read_split_csv(data_config["split_files"]["train"])
    validation_rows = read_split_csv(data_config["split_files"]["validation"])
    expected_sizes = data_config.get("expected_sizes", {})
    if len(train_rows) != expected_sizes.get("train"):
        raise ValueError(
            f"Train split has {len(train_rows)} samples; "
            f"expected {expected_sizes.get('train')}."
        )
    if len(validation_rows) != expected_sizes.get("validation"):
        raise ValueError(
            f"Validation split has {len(validation_rows)} samples; "
            f"expected {expected_sizes.get('validation')}."
        )

    classes = sorted({row["class_name"] for row in train_rows})
    if len(classes) != experiment["class_count"]:
        raise ValueError(
            f"Expected {experiment['class_count']} train classes, found {len(classes)}."
        )
    if any(row["class_name"] not in classes for row in validation_rows):
        raise ValueError("Validation split contains a class absent from train.")
    class_to_idx = {name: index for index, name in enumerate(classes)}

    dataset_env = data_config["dataset_root_env"]
    if dataset_env not in os.environ:
        raise RuntimeError(f"{dataset_env} environment variable is not set.")
    dataset_root = os.environ[dataset_env]

    print("Classes:", classes)
    print("Training samples:", len(train_rows))
    print("Validation samples:", len(validation_rows))
    print("Dataset root:", dataset_root)

    image_transforms = build_transforms(experiment["image_size"])
    train_dataset = TomatoLeafDataset(
        train_rows,
        dataset_root,
        class_to_idx,
        image_transforms["train"],
    )
    validation_dataset = TomatoLeafDataset(
        validation_rows,
        dataset_root,
        class_to_idx,
        image_transforms["validation"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=experiment["batch_size"],
        shuffle=True,
        num_workers=runtime["num_workers"],
        pin_memory=runtime["pin_memory"],
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=experiment["batch_size"],
        shuffle=False,
        num_workers=runtime["num_workers"],
        pin_memory=runtime["pin_memory"],
    )

    model = EfficientNetB0DynamicAttention(
        num_classes=experiment["class_count"],
        pretrained=experiment["pretrained"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment["learning_rate"],
    )

    output_dir = Path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(output_dir, config)

    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    epochs = experiment["epochs"]

    print("\nStarting training...")
    print("Model:", experiment["model_name"])
    print("Epochs:", epochs)
    print("Batch size:", experiment["batch_size"])
    print("Learning rate:", experiment["learning_rate"])
    print("Output directory:", output_dir)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        validation_metrics = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
            classes,
        )

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Validation Loss: {validation_metrics['loss']:.6f}")
        print(f"Validation Accuracy: {validation_metrics['accuracy']:.6f}")
        print(
            "Validation Macro Precision: "
            f"{validation_metrics['macro_precision']:.6f}"
        )
        print(
            "Validation Macro Recall: "
            f"{validation_metrics['macro_recall']:.6f}"
        )
        print(f"Validation Macro F1: {validation_metrics['macro_f1']:.6f}")
        print(f"Validation Weighted F1: {validation_metrics['weighted_f1']:.6f}")

        is_new_best = validation_metrics["macro_f1"] > best_val_f1
        if is_new_best:
            best_val_f1 = validation_metrics["macro_f1"]
            save_checkpoint(
                output_dir / "best_checkpoint.pt",
                model,
                optimizer,
                epoch,
                config,
            )
            print(
                "New best validation Macro F1 achieved: "
                f"{best_val_f1:.6f}. Best checkpoint saved."
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": validation_metrics,
                "best_validation_macro_f1": best_val_f1,
                "is_new_best": is_new_best,
            }
        )
        save_training_history(output_dir, history)

    print("\nTraining completed.")
    print("Best validation Macro F1:", best_val_f1)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()

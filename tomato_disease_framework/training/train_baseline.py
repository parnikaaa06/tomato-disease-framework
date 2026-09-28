"""Training and validation loops for the EfficientNetB0 Phase 3 baseline."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.classification_metrics import classification_metrics


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
            raise RuntimeError(
                "CUDA was requested but is not available."
            )
        return torch.device("cuda")

    if requested_device == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
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
    model,
    loader,
    criterion,
    device,
    class_names,
):
    """Validate the model for one epoch."""
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )
        all_targets.extend(
            labels.cpu().numpy().tolist()
        )

    metrics = classification_metrics(
        all_targets,
        all_predictions,
        class_names,
    )

    metrics["loss"] = (
        total_loss / len(loader.dataset)
    )

    return metrics


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    epoch: int,
    config: dict,
) -> None:
    """Save model and optimizer checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }

    torch.save(checkpoint, path)


def save_experiment_config(
    output_dir: str | Path,
    config: dict,
) -> None:
    """Save experiment configuration."""
    output = Path(output_dir)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run EfficientNetB0 baseline training."""

    import argparse
    import yaml

    from preprocessing.dataset import (
        load_phase2_splits,
        TomatoLeafDataset,
    )
    from preprocessing.transforms import (
        build_transforms,
    )
    from models.efficientnet_baseline import (
        build_efficientnet_b0,
    )

    # ---------------------------------------------------------
    # Arguments
    # ---------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to Phase 3 configuration YAML.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    experiment = config["experiment"]
    data_config = config["data"]
    runtime = config["runtime"]

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    set_reproducibility(
        experiment["seed"]
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = resolve_device(config)

    print("Configuration loaded.")
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ---------------------------------------------------------
    # Phase 2 splits
    # ---------------------------------------------------------

    split_dir = data_config["splits_dir"]

    splits = load_phase2_splits(
        split_dir
    )

    classes = sorted(
        {
            row["class_name"]
            for row in splits["train"]
        }
    )

    class_to_idx = {
        name: i
        for i, name in enumerate(classes)
    }

    print(
        "Classes:",
        classes,
    )

    print(
        "Training samples:",
        len(splits["train"]),
    )

    print(
        "Validation samples:",
        len(splits["validation"]),
    )

    # ---------------------------------------------------------
    # Dataset root
    # ---------------------------------------------------------

    dataset_env = data_config[
        "dataset_root_env"
    ]

    if dataset_env not in os.environ:
        raise RuntimeError(
            f"{dataset_env} environment variable "
            "is not set."
        )

    dataset_root = os.environ[
        dataset_env
    ]

    print(
        "Dataset root:",
        dataset_root,
    )

    # ---------------------------------------------------------
    # Transforms
    # ---------------------------------------------------------

    transforms = build_transforms(
        experiment["image_size"]
    )

    # ---------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------

    train_dataset = TomatoLeafDataset(
        splits["train"],
        dataset_root,
        class_to_idx,
        transforms["train"],
    )

    validation_dataset = TomatoLeafDataset(
        splits["validation"],
        dataset_root,
        class_to_idx,
        transforms["validation"],
    )

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=experiment[
            "batch_size"
        ],
        shuffle=True,
        num_workers=runtime[
            "num_workers"
        ],
        pin_memory=runtime[
            "pin_memory"
        ],
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=experiment[
            "batch_size"
        ],
        shuffle=False,
        num_workers=runtime[
            "num_workers"
        ],
        pin_memory=runtime[
            "pin_memory"
        ],
    )

    print(
        "Train batches:",
        len(train_loader),
    )

    print(
        "Validation batches:",
        len(validation_loader),
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = build_efficientnet_b0(
        pretrained=experiment[
            "pretrained"
        ]
    ).to(device)

    print(
        "Model:",
        experiment["model_name"],
    )

    print(
        "Number of classes:",
        experiment["class_count"],
    )

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment[
            "learning_rate"
        ],
    )

    # ---------------------------------------------------------
    # Training settings
    # ---------------------------------------------------------

    epochs = experiment["epochs"]

    output_dir = Path(
        experiment["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_f1 = -1.0

    print("\nStarting training...")
    print(
        "Epochs:",
        epochs,
    )

    print(
        "Batch size:",
        experiment["batch_size"],
    )

    print(
        "Learning rate:",
        experiment["learning_rate"],
    )

    print(
        "Output directory:",
        output_dir,
    )

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_metrics = (
            validate_one_epoch(
                model,
                validation_loader,
                criterion,
                device,
                classes,
            )
        )

        print(
            f"\nEpoch {epoch}/{epochs}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_metrics['loss']:.6f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_metrics['accuracy']:.6f}"
        )

        print(
            f"Validation Macro F1: "
            f"{validation_metrics['macro_f1']:.6f}"
        )

        # -----------------------------------------------------
        # Best checkpoint
        # -----------------------------------------------------

        if (
            validation_metrics["macro_f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                validation_metrics[
                    "macro_f1"
                ]
            )

            save_checkpoint(
                output_dir
                / "best_checkpoint.pt",
                model,
                optimizer,
                epoch,
                config,
            )

            print(
                "Best checkpoint saved."
            )

    # ---------------------------------------------------------
    # Save configuration
    # ---------------------------------------------------------

    save_experiment_config(
        output_dir,
        config,
    )

    print(
        "\nTraining completed."
    )

    print(
        "Best validation Macro F1:",
        best_val_f1,
    )

    print(
        "Output directory:",
        output_dir,
    )


if __name__ == "__main__":
    main()
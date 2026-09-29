"""Evaluate the trained dynamic-attention classifier on the official test split."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from evaluation.classification_metrics import classification_metrics
from models.efficientnet_dynamic_attention import EfficientNetB0DynamicAttention
from preprocessing.dataset import TomatoLeafDataset, read_split_csv
from preprocessing.transforms import build_transforms


EXPECTED_TEST_SIZE = 2725


def resolve_device(config: dict[str, Any]) -> torch.device:
    """Resolve the evaluation device from configuration."""
    requested_device = config["runtime"].get("device", "auto")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if requested_device == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    """Run inference and calculate classification metrics on a split."""
    model.eval()
    all_predictions: list[int] = []
    all_targets: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        all_predictions.extend(predictions.cpu().tolist())
        all_targets.extend(labels.tolist())

    metrics = classification_metrics(
        all_targets,
        all_predictions,
        class_names,
    )
    _, _, _, supports = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    for index, class_name in enumerate(class_names):
        metrics["per_class"][class_name]["support"] = int(supports[index])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/phase3_attention.yaml",
        help="Path to the Phase 3 attention configuration YAML.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Attention evaluation config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    experiment = config["experiment"]
    data_config = config["data"]
    runtime = config["runtime"]

    dataset_env = data_config["dataset_root_env"]
    dataset_root = os.environ.get(dataset_env)
    if not dataset_root:
        raise RuntimeError(
            f"Dataset root environment variable {dataset_env} is not set or is empty."
        )
    if not Path(dataset_root).is_dir():
        raise FileNotFoundError(
            f"Dataset root from {dataset_env} is not a directory: {dataset_root}"
        )

    test_split_path = Path(data_config["split_files"]["test"])
    if not test_split_path.is_file():
        raise FileNotFoundError(
            f"Official test split CSV not found: {test_split_path}"
        )
    test_rows = read_split_csv(test_split_path)
    if len(test_rows) != EXPECTED_TEST_SIZE:
        raise ValueError(
            f"Official test split has {len(test_rows)} samples; "
            f"expected exactly {EXPECTED_TEST_SIZE}."
        )

    class_names = sorted({row["class_name"] for row in test_rows})
    if len(class_names) != experiment["class_count"]:
        raise ValueError(
            f"Test split contains {len(class_names)} classes; "
            f"expected {experiment['class_count']}."
        )
    class_to_index = {name: index for index, name in enumerate(class_names)}
    device = resolve_device(config)

    test_transform = build_transforms(experiment["image_size"])["test"]
    test_dataset = TomatoLeafDataset(
        test_rows,
        dataset_root,
        class_to_index,
        test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=experiment["batch_size"],
        shuffle=False,
        num_workers=runtime["num_workers"],
        pin_memory=runtime["pin_memory"],
    )

    checkpoint_path = (
        Path(experiment["output_dir"]) / "best_checkpoint.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best attention checkpoint not found: {checkpoint_path}"
        )

    # The checkpoint supplies the trained backbone weights; avoid downloading
    # ImageNet weights again when constructing the matching model architecture.
    model = EfficientNetB0DynamicAttention(
        num_classes=experiment["class_count"],
        pretrained=False,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain 'model_state_dict': {checkpoint_path}"
        )
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Configuration:", config_path)
    print("Device:", device)
    print("Dataset root:", dataset_root)
    print("Test split:", test_split_path)
    print("Test samples:", len(test_dataset))
    print("Checkpoint:", checkpoint_path)

    metrics = evaluate(model, test_loader, device, class_names)
    results = {
        "experiment": experiment["name"],
        "model_name": experiment["model_name"],
        "checkpoint": str(checkpoint_path),
        "test_split": str(test_split_path),
        "test_sample_count": len(test_dataset),
        "class_names": class_names,
        "metrics": metrics,
    }

    results_path = Path(experiment["output_dir"]) / "test_results.yaml"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as results_file:
        yaml.safe_dump(results, results_file, sort_keys=False, allow_unicode=True)

    print("\nTest metrics")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Macro Precision: {metrics['macro_precision']:.6f}")
    print(f"Macro Recall: {metrics['macro_recall']:.6f}")
    print(f"Macro F1: {metrics['macro_f1']:.6f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.6f}")
    print("\nPer-class metrics")
    for class_name in class_names:
        class_metrics = metrics["per_class"][class_name]
        print(
            f"{class_name}: precision={class_metrics['precision']:.6f}, "
            f"recall={class_metrics['recall']:.6f}, "
            f"f1={class_metrics['f1']:.6f}, "
            f"support={class_metrics['support']}"
        )

    print("\nConfusion Matrix (rows=true, columns=predicted)")
    print("Class order:", class_names)
    for row in metrics["confusion_matrix"]:
        print(row)
    print("\nResults saved to:", results_path)


if __name__ == "__main__":
    main()
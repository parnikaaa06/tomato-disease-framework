"""Evaluate Grad-CAM on a deterministic balanced sample of the official test split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image
from torch import nn

from evaluation.visualize_gradcam import load_checkpoint, resolve_device
from explainability.gradcam import generate_gradcam
from models.efficientnet_baseline import build_efficientnet_b0
from preprocessing.dataset import TomatoLeafDataset, read_split_csv
from preprocessing.transforms import build_transforms


EXPECTED_TEST_SIZE = 2725
EXPECTED_CLASS_COUNT = 10
RESULT_COLUMNS = (
    "image_path",
    "true_class",
    "predicted_class",
    "correctness",
    "heatmap_path",
)


def validate_test_split(
    rows: list[dict[str, str]],
    configured_class_count: int,
) -> list[str]:
    """Validate the official test split size and its class mapping."""
    if len(rows) != EXPECTED_TEST_SIZE:
        raise ValueError(
            f"Official test split has {len(rows)} rows; "
            f"expected exactly {EXPECTED_TEST_SIZE}."
        )

    class_names = sorted({row["class_name"] for row in rows})
    if len(class_names) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Official test split contains {len(class_names)} classes; "
            f"expected exactly {EXPECTED_CLASS_COUNT}."
        )
    if configured_class_count != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Configured class_count is {configured_class_count}; "
            f"expected {EXPECTED_CLASS_COUNT}."
        )
    if len(class_names) != configured_class_count:
        raise ValueError(
            f"Official test split contains {len(class_names)} classes; "
            f"configured class_count is {configured_class_count}."
        )
    return class_names


def select_balanced_samples(
    rows: list[dict[str, str]],
    samples_per_class: int = 5,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Select a reproducible, without-replacement sample per test class."""
    if isinstance(samples_per_class, bool) or not isinstance(samples_per_class, int):
        raise ValueError("samples_per_class must be a positive integer.")
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be a positive integer.")

    grouped: dict[str, list[dict[str, str]]] = {}
    seen_image_paths: set[str] = set()
    for row in rows:
        image_path = row["image_path"]
        if image_path in seen_image_paths:
            raise ValueError(
                f"Test split contains duplicate image path: {image_path}"
            )
        seen_image_paths.add(image_path)
        grouped.setdefault(row["class_name"], []).append(row)
    if not grouped:
        raise ValueError("Cannot select samples from an empty test split.")

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for class_name in sorted(grouped):
        class_rows = grouped[class_name]
        if len(class_rows) < samples_per_class:
            raise ValueError(
                f"Class {class_name!r} has {len(class_rows)} test images; "
                f"cannot select {samples_per_class} without replacement."
            )
        selected.extend(rng.sample(class_rows, samples_per_class))
    return selected


def _safe_filename_part(value: str) -> str:
    """Convert a class or image name into a filesystem-safe filename segment."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return (safe or "image")[:80]


def _save_visualization(
    original: Image.Image,
    heatmap: torch.Tensor,
    true_class: str,
    predicted_class: str,
    output_path: Path,
) -> None:
    """Write one original/heatmap/overlay visualization and close its figure."""
    heatmap_image = heatmap.detach().cpu().numpy()
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original image")
    axes[1].imshow(heatmap_image, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heatmap (not a lesion mask)")
    axes[2].imshow(original)
    axes[2].imshow(heatmap_image, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Grad-CAM overlay")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"True: {true_class} | Predicted: {predicted_class}")
    figure.tight_layout()
    try:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)


def evaluate_selected_samples(
    model: nn.Module,
    selected_rows: list[dict[str, str]],
    dataset_root: str | Path,
    class_names: list[str],
    image_size: int,
    device: torch.device,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Generate per-image predictions and Grad-CAM visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    class_to_index = {
        class_name: index for index, class_name in enumerate(class_names)
    }
    dataset = TomatoLeafDataset(
        selected_rows,
        dataset_root,
        class_to_index,
        build_transforms(image_size)["test"],
    )

    records: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        image_tensor, true_class_index = dataset[index]
        image_path = Path(dataset_root) / row["image_path"]
        with Image.open(image_path) as image_file:
            original = image_file.convert("RGB")

        heatmap, predicted_index = generate_gradcam(
            model=model,
            input_tensor=image_tensor.unsqueeze(0).to(device),
            target_layer=model.features[-1],
        )
        true_class = class_names[true_class_index]
        predicted_class = class_names[predicted_index]
        safe_class = _safe_filename_part(true_class)
        safe_stem = _safe_filename_part(Path(row["image_path"]).stem)
        heatmap_path = output_path / (
            f"{index + 1:04d}_{safe_class}_{safe_stem}_gradcam.png"
        )
        _save_visualization(
            original,
            heatmap,
            true_class,
            predicted_class,
            heatmap_path,
        )
        records.append(
            {
                "image_path": row["image_path"],
                "true_class": true_class,
                "predicted_class": predicted_class,
                "correctness": true_class == predicted_class,
                "heatmap_path": str(heatmap_path.resolve()),
            }
        )
    return records


def write_evaluation_outputs(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    checkpoint_path: str | Path,
    test_split_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write per-image CSV and machine-readable balanced-sample summary JSON."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "gradcam_evaluation.csv"
    summary_path = directory / "gradcam_evaluation_summary.json"

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    per_class_counts: dict[str, int] = {}
    for record in records:
        class_name = record["true_class"]
        per_class_counts[class_name] = per_class_counts.get(class_name, 0) + 1

    sample_size = len(records)
    correct_predictions = sum(bool(record["correctness"]) for record in records)
    summary: dict[str, Any] = {
        "sample_size": sample_size,
        "per_class_counts": dict(sorted(per_class_counts.items())),
        "correct_predictions": correct_predictions,
        "accuracy": correct_predictions / sample_size if sample_size else 0.0,
        "accuracy_scope": "balanced sample only; not the full test set",
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "test_split_path": str(Path(test_split_path).expanduser().resolve()),
    }
    with summary_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
        json_file.write("\n")

    return csv_path, summary_path, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Grad-CAM on a deterministic balanced sample of the "
            "official Phase 2 test split."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase3_baseline.yaml"),
        help="Phase 3 baseline YAML (default: configs/phase3_baseline.yaml).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="PlantVillage image root; defaults to the configured environment variable.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Explicit path to the existing Phase 3 best_checkpoint.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase8/gradcam_evaluation"),
        help="Directory for visualizations, CSV, and JSON (default: results/phase8/gradcam_evaluation).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=5,
        help="Number of official test images sampled per class (default: 5).",
    )
    args = parser.parse_args(argv)

    config_path = args.config.expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Baseline configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Baseline configuration is not a YAML mapping: {config_path}")

    experiment = config["experiment"]
    data_config = config["data"]
    dataset_env = data_config["dataset_root_env"]
    dataset_root_value = args.dataset_root or os.environ.get(dataset_env)
    if not dataset_root_value:
        raise RuntimeError(
            f"Provide --dataset-root or set {dataset_env} to the PlantVillage image root."
        )
    dataset_root = Path(dataset_root_value).expanduser()
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"PlantVillage image root is not a directory: {dataset_root}"
        )

    test_split_path = Path(data_config["split_files"]["test"]).expanduser()
    rows = read_split_csv(test_split_path)
    class_names = validate_test_split(rows, experiment["class_count"])
    selected_rows = select_balanced_samples(
        rows,
        samples_per_class=args.samples_per_class,
        seed=experiment.get("seed", 42),
    )

    device = resolve_device(config.get("runtime", {}))
    model = build_efficientnet_b0(
        num_classes=experiment["class_count"],
        pretrained=False,
    ).to(device)
    checkpoint_path = args.checkpoint.expanduser()
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    output_dir = args.output_dir.expanduser()
    records = evaluate_selected_samples(
        model=model,
        selected_rows=selected_rows,
        dataset_root=dataset_root,
        class_names=class_names,
        image_size=experiment["image_size"],
        device=device,
        output_dir=output_dir,
    )
    csv_path, summary_path, summary = write_evaluation_outputs(
        records,
        output_dir,
        checkpoint_path,
        test_split_path,
    )

    print(f"Balanced sample total: {summary['sample_size']}")
    print(f"Correct predictions: {summary['correct_predictions']}")
    print(
        "Balanced sample accuracy "
        "(sample only; not the full test set): "
        f"{summary['accuracy']:.4f}"
    )
    print("Visualization and metadata directory:", output_dir.resolve())
    print("Per-image results CSV:", csv_path.resolve())
    print("Machine-readable summary JSON:", summary_path.resolve())


if __name__ == "__main__":
    main()

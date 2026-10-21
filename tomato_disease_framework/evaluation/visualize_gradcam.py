"""Generate a Grad-CAM overlay using the trained Phase 3 baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image
from torch import nn

from explainability.gradcam import generate_gradcam
from models.efficientnet_baseline import build_efficientnet_b0
from preprocessing.dataset import TomatoLeafDataset, read_split_csv
from preprocessing.transforms import build_transforms


EXPECTED_TEST_SIZE = 2725


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    """Load the Phase 3 model weights from its expected checkpoint format."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain 'model_state_dict': {path}"
        )
    if not isinstance(checkpoint["model_state_dict"], dict):
        raise ValueError(f"Checkpoint model_state_dict is invalid: {path}")
    model.load_state_dict(checkpoint["model_state_dict"])


def resolve_device(runtime_config: dict[str, Any]) -> torch.device:
    """Resolve the configured inference device."""
    requested_device = runtime_config.get("device", "auto")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device != "auto":
        raise ValueError(
            f"Unsupported runtime device setting: {requested_device!r}"
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Grad-CAM visualization for one image from the official "
            "Phase 2 test split."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase3_baseline.yaml"),
        help="Path to the Phase 3 baseline configuration YAML.",
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
        default=None,
        help="Phase 3 best_checkpoint.pt path; defaults to the configured output_dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase8/gradcam_example.png"),
        help="Output PNG path; can point to a mounted Google Drive directory.",
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
    runtime_config = config.get("runtime", {})

    dataset_env = data_config["dataset_root_env"]
    dataset_root_value = args.dataset_root or os.environ.get(dataset_env)
    if not dataset_root_value:
        raise RuntimeError(
            f"Provide --dataset-root or set {dataset_env} to the PlantVillage "
            "image root."
        )
    dataset_root = Path(dataset_root_value).expanduser()
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"PlantVillage image root is not a directory: {dataset_root}"
        )

    test_split_path = Path(data_config["split_files"]["test"]).expanduser()
    rows = read_split_csv(test_split_path)
    if len(rows) != EXPECTED_TEST_SIZE:
        raise ValueError(
            f"Official test split has {len(rows)} rows; "
            f"expected exactly {EXPECTED_TEST_SIZE}."
        )
    class_names = sorted({row["class_name"] for row in rows})
    if len(class_names) != experiment["class_count"]:
        raise ValueError(
            f"Official test split contains {len(class_names)} classes; "
            f"expected {experiment['class_count']}."
        )
    class_to_index = {name: index for index, name in enumerate(class_names)}

    device = resolve_device(runtime_config)
    model = build_efficientnet_b0(
        num_classes=experiment["class_count"],
        pretrained=False,
    ).to(device)
    checkpoint_path = args.checkpoint or (
        Path(experiment["output_dir"]) / "best_checkpoint.pt"
    )
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    test_dataset = TomatoLeafDataset(
        rows,
        dataset_root,
        class_to_index,
        build_transforms(experiment["image_size"])["test"],
    )
    row = rows[0]
    input_tensor, true_class_index = test_dataset[0]
    input_tensor = input_tensor.unsqueeze(0).to(device)
    image_path = dataset_root / row["image_path"]
    with Image.open(image_path) as image_file:
        original = image_file.convert("RGB")

    heatmap, predicted_idx = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        target_layer=model.features[-1],
    )
    heatmap_image = heatmap.numpy()

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original image")
    axes[1].imshow(heatmap_image, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heatmap (not a lesion mask)")
    axes[2].imshow(original)
    axes[2].imshow(
        heatmap_image,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Grad-CAM overlay")
    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"True: {class_names[true_class_index]} | "
        f"Predicted: {class_names[predicted_idx]}"
    )
    figure.tight_layout()
    try:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)

    print("Image:", image_path)
    print("True class:", class_names[true_class_index])
    print("Predicted class:", class_names[predicted_idx])
    print("Checkpoint:", checkpoint_path)
    print("Grad-CAM output:", output_path.resolve())


if __name__ == "__main__":
    main()

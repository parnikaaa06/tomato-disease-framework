"""Run one official test image through the Phase 3, 8, and 9 pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from evaluation.evaluate_gradcam import (
    EXPECTED_TEST_SIZE,
    validate_test_split,
)
from evaluation.visualize_gradcam import load_checkpoint, resolve_device
from models.efficientnet_baseline import build_efficientnet_b0
from preprocessing.dataset import TomatoLeafDataset, read_split_csv
from preprocessing.transforms import build_transforms
from vlm.explanation import (
    MODEL_ID,
    ClassifierResult,
    ExplanationBackend,
    ExplanationRequest,
    generate_structured_explanation,
)
from vlm.gemini_backend import GeminiBackend
from vlm.schema import validate_output


def _save_experiment_visualization(
    original: Image.Image,
    heatmap: torch.Tensor,
    true_class: str,
    predicted_class: str,
    output_path: Path,
) -> None:
    heatmap_array = heatmap.detach().cpu().numpy()
    figure = Figure(figsize=(14, 4))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    axes[0].imshow(original)
    axes[0].set_title("Original image")
    axes[1].imshow(heatmap_array, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heatmap (not a lesion mask)")
    axes[2].imshow(original)
    axes[2].imshow(heatmap_array, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Grad-CAM overlay")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"True: {true_class} | Predicted: {predicted_class}")
    figure.tight_layout()
    try:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    finally:
        figure.clear()


def _row_image_path(dataset_root: Path, row: dict[str, str]) -> Path:
    root = dataset_root.resolve()
    image_path = (root / row["image_path"]).resolve()
    try:
        image_path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Test split image path escapes the dataset root: {row['image_path']}"
        ) from error
    return image_path


def select_official_test_row(
    rows: list[dict[str, str]],
    dataset_root: str | Path,
    *,
    image_path: str | Path | None = None,
    row_index: int | None = None,
) -> tuple[int, dict[str, str]]:
    """Choose one row by zero-based test-CSV index or a path listed in that CSV."""
    if image_path is not None and row_index is not None:
        raise ValueError("Specify either image_path or row_index, not both.")

    root = Path(dataset_root).expanduser().resolve()
    if image_path is not None:
        requested = Path(image_path).expanduser()
        if not requested.is_absolute():
            requested = root / requested
        requested = requested.resolve()
        for index, row in enumerate(rows):
            if _row_image_path(root, row) == requested:
                return index, row
        raise ValueError(
            f"Image is not listed in the official test split: {image_path}"
        )

    selected_index = 0 if row_index is None else row_index
    if isinstance(selected_index, bool) or not isinstance(selected_index, int):
        raise ValueError("row_index must be a zero-based integer.")
    if not 0 <= selected_index < len(rows):
        raise ValueError(
            f"row_index {selected_index} is outside the test split "
            f"(0 to {len(rows) - 1})."
        )
    return selected_index, rows[selected_index]


class _CapturingBackend:
    """Capture application-owned classifier and Grad-CAM values before VLM call."""

    def __init__(
        self,
        backend: ExplanationBackend | None,
        backend_factory: Callable[[], ExplanationBackend],
    ) -> None:
        self.backend = backend
        self.backend_factory = backend_factory
        self.owns_backend = backend is None
        self.classifier: ClassifierResult | None = None
        self.heatmap: torch.Tensor | None = None

    def explain(self, request: ExplanationRequest) -> dict[str, Any]:
        self.classifier = request.classifier
        self.heatmap = request.gradcam_heatmap.detach().cpu().clone()
        if self.backend is None:
            self.backend = self.backend_factory()
        return self.backend.explain(request)

    def close(self) -> None:
        if self.owns_backend and self.backend is not None:
            close = getattr(self.backend, "close", None)
            if callable(close):
                close()


def run_single_image_experiment(
    *,
    config_path: str | Path,
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    backend: ExplanationBackend | None = None,
    backend_factory: Callable[[], ExplanationBackend] = GeminiBackend,
    image_path: str | Path | None = None,
    row_index: int | None = None,
) -> dict[str, Any]:
    """Run an image confirmed in the configured Phase 2 test split."""
    config_file_path = Path(config_path).expanduser()
    if not config_file_path.is_file():
        raise FileNotFoundError(f"Baseline configuration not found: {config_file_path}")
    with config_file_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Baseline configuration is not a YAML mapping: {config_file_path}")

    experiment = config["experiment"]
    data_config = config["data"]
    class_count = experiment["class_count"]
    data_root = Path(dataset_root).expanduser()
    if not data_root.is_dir():
        raise FileNotFoundError(f"PlantVillage image root is not a directory: {data_root}")

    test_split_path = Path(data_config["split_files"]["test"]).expanduser()
    rows = read_split_csv(test_split_path)
    class_names = validate_test_split(rows, class_count)
    if len(rows) != EXPECTED_TEST_SIZE:
        raise ValueError(
            f"Official test split has {len(rows)} rows; "
            f"expected exactly {EXPECTED_TEST_SIZE}."
        )
    selected_index, selected_row = select_official_test_row(
        rows,
        data_root,
        image_path=image_path,
        row_index=row_index,
    )

    checkpoint = Path(checkpoint_path).expanduser()
    device = resolve_device(config.get("runtime", {}))
    model = build_efficientnet_b0(num_classes=class_count, pretrained=False).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    class_to_index = {name: index for index, name in enumerate(class_names)}
    dataset = TomatoLeafDataset(
        [selected_row],
        data_root,
        class_to_index,
        build_transforms(experiment["image_size"])["test"],
    )
    input_tensor, true_class_index = dataset[0]
    source_image_path = _row_image_path(data_root, selected_row)
    with Image.open(source_image_path) as source_image:
        original_image = source_image.convert("RGB")

    result_dir = Path(output_dir).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)
    image_stem = re.sub(
        r"[^A-Za-z0-9_-]+", "_", Path(selected_row["image_path"]).stem
    ).strip("_") or "image"
    image_id = f"test-row-{selected_index:04d}-{image_stem[:80]}"
    visualization_path = result_dir / f"{image_id}_gradcam.png"
    result_path = result_dir / f"{image_id}_phase9_result.json"

    capturing_backend = _CapturingBackend(backend, backend_factory)
    validated_explanation: dict[str, Any] | None = None
    failure: dict[str, str] | None = None
    try:
        validated_explanation = generate_structured_explanation(
            model=model,
            input_tensor=input_tensor.unsqueeze(0).to(device),
            image=original_image,
            class_names=class_names,
            backend=capturing_backend,
        )
        if capturing_backend.classifier is None:
            raise RuntimeError("Explanation orchestration did not capture classifier output.")
        expected_classifier = asdict(capturing_backend.classifier)
        validate_output(
            validated_explanation,
            expected_class=expected_classifier["predicted_class"],
            expected_class_index=expected_classifier["class_index"],
        )
        if validated_explanation["classifier"] != expected_classifier:
            raise RuntimeError("Classifier result changed in the validated explanation.")
    except Exception as error:
        failure = {
            "stage": "structured_explanation",
            "error_type": type(error).__name__,
            "message": "No validated VLM explanation was produced.",
        }
    finally:
        try:
            capturing_backend.close()
        except Exception as error:
            if failure is None:
                validated_explanation = None
                failure = {
                    "stage": "backend_cleanup",
                    "error_type": type(error).__name__,
                    "message": "The Gemini client could not be closed cleanly.",
                }
            else:
                failure["cleanup_error_type"] = type(error).__name__

    if capturing_backend.classifier is None or capturing_backend.heatmap is None:
        raise RuntimeError(
            "Classifier or Grad-CAM evidence was not produced; no experiment "
            "result can be recorded."
        )

    classifier_result = asdict(capturing_backend.classifier)
    true_class = class_names[true_class_index]
    _save_experiment_visualization(
        original_image,
        capturing_backend.heatmap,
        true_class,
        classifier_result["predicted_class"],
        visualization_path,
    )
    result: dict[str, Any] = {
        "status": "completed" if failure is None else "explanation_failed",
        "image": {
            "identifier": image_id,
            "official_test_split_row_index": selected_index,
            "path": str(source_image_path.resolve()),
            "true_class": true_class,
        },
        "checkpoint": {
            "identifier": checkpoint.name,
            "path": str(checkpoint.resolve()),
        },
        "model": {
            "name": experiment["model_name"],
            "model_id": MODEL_ID,
        },
        "prediction": {
            **classifier_result,
            "correct": classifier_result["predicted_class"] == true_class,
        },
        "test_split_path": str(test_split_path.resolve()),
        "gradcam_visualization_path": str(visualization_path.resolve()),
        "validated_explanation": validated_explanation,
        "failure_status": failure,
        "scope_note": (
            "Single official test image only. This experiment does not establish "
            "disease severity, progression, or field reliability. Grad-CAM is "
            "model attribution evidence, not lesion ground truth."
        ),
    }
    with result_path.open("w", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2, ensure_ascii=False, allow_nan=False)
        result_file.write("\n")
    result["result_json_path"] = str(result_path.resolve())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one official Phase 2 test image through the Phase 3 classifier, "
            "Phase 8 Grad-CAM, and Phase 9 Gemini explanation."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase3_baseline.yaml"),
        help="Phase 3 config YAML (default: configs/phase3_baseline.yaml).",
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
        help="Path to the existing Phase 3 best_checkpoint.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the JSON experiment result and Grad-CAM visualization.",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Zero-based row in the official test CSV (default: 0).",
    )
    selector.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Image path listed in the official test CSV; relative paths use dataset root.",
    )
    args = parser.parse_args(argv)

    data_config: dict[str, Any]
    config_path = args.config.expanduser()
    if not config_path.is_file():
        parser.error(f"Phase 3 config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        parser.error(f"Invalid Phase 3 config mapping: {config_path}")
    data_config = config["data"]
    dataset_value = args.dataset_root or os.environ.get(data_config["dataset_root_env"])
    if not dataset_value:
        parser.error(
            f"Provide --dataset-root or set {data_config['dataset_root_env']}."
        )

    result = run_single_image_experiment(
        config_path=config_path,
        dataset_root=dataset_value,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        image_path=args.image_path,
        row_index=args.row_index,
    )

    print("Experiment status:", result["status"])
    print("Image:", result["image"]["path"])
    print("True class:", result["image"]["true_class"])
    print("Predicted class:", result["prediction"]["predicted_class"])
    print(
        "Uncalibrated softmax score:",
        result["prediction"]["softmax_score"],
    )
    print("Grad-CAM visualization:", result["gradcam_visualization_path"])
    print("Experiment result JSON:", result["result_json_path"])
    if result["failure_status"] is not None:
        print("Structured explanation failed; no explanation was fabricated.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

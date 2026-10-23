from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

import evaluation.run_phase9_experiment as phase9_experiment
from vlm.explanation import ExplanationRequest
from vlm.gemini_backend import GeminiBackendError


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 10)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.features(image))
        return self.classifier(self.pool(features).flatten(1))


class FakeBackend:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.requests: list[ExplanationRequest] = []

    def explain(self, request: ExplanationRequest) -> dict[str, Any]:
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return {
            "observations": [
                {
                    "text": "A leaf is visible in the input image.",
                    "evidence_sources": ["input_image"],
                    "certainty": "direct_visual_observation",
                }
            ],
            "summary": "The classifier prediction is not a confirmed diagnosis.",
            "claims": [
                {
                    "text": "The heatmap indicates relative model attribution.",
                    "claim_type": "visual_observation",
                    "evidence_sources": ["gradcam_heatmap"],
                    "certainty": "supported",
                }
            ],
        }


def _make_test_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    rows = [
        {
            "image_path": f"images/class_{index % 10}/image_{index}.png",
            "class_name": f"Tomato___class_{index % 10}",
        }
        for index in range(phase9_experiment.EXPECTED_TEST_SIZE)
    ]
    image_relative_path = rows[0]["image_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "class_name"])
        writer.writeheader()
        writer.writerows(rows)
    return rows, image_relative_path


def _make_config(path: Path, test_split_path: Path) -> None:
    path.write_text(
        (
            "experiment:\n"
            "  model_name: EfficientNetB0\n"
            "  class_count: 10\n"
            "  image_size: 32\n"
            "data:\n"
            "  split_files:\n"
            f"    test: {test_split_path.as_posix()}\n"
            "runtime:\n"
            "  device: cpu\n"
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("row_index", [None, 0, 2])
def test_select_official_test_row_by_deterministic_index(
    tmp_path: Path,
    row_index: int | None,
):
    rows = [
        {"image_path": f"image-{index}.png", "class_name": "Tomato___healthy"}
        for index in range(3)
    ]
    selector_args = {} if row_index is None else {"row_index": row_index}
    selected_index, selected = phase9_experiment.select_official_test_row(
        rows,
        tmp_path,
        **selector_args,
    )

    assert selected_index == (0 if row_index is None else row_index)
    assert selected == rows[selected_index]


def test_select_official_test_row_requires_membership_and_valid_index(tmp_path: Path):
    rows = [{"image_path": "official.png", "class_name": "Tomato___healthy"}]

    with pytest.raises(ValueError, match="not listed"):
        phase9_experiment.select_official_test_row(
            rows,
            tmp_path,
            image_path="other.png",
        )
    with pytest.raises(ValueError, match="outside the test split"):
        phase9_experiment.select_official_test_row(rows, tmp_path, row_index=1)
    with pytest.raises(ValueError, match="either image_path or row_index"):
        phase9_experiment.select_official_test_row(
            rows,
            tmp_path,
            image_path="official.png",
            row_index=0,
        )
    with pytest.raises(ValueError, match="escapes the dataset root"):
        phase9_experiment.select_official_test_row(
            [{"image_path": "../outside.png", "class_name": "Tomato___healthy"}],
            tmp_path,
            image_path="outside.png",
        )


@pytest.mark.parametrize(
    ("failure_mode", "backend_failure"),
    [
        (None, None),
        (
            "safe_backend",
            GeminiBackendError("Gemini service error (HTTP 504)."),
        ),
        (
            "unexpected_backend",
            RuntimeError("raw SDK response and secret test-token"),
        ),
        ("initialization", None),
    ],
)
def test_single_image_experiment_saves_valid_result_or_explicit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str | None,
    backend_failure: Exception | None,
):
    dataset_root = tmp_path / "dataset"
    split_path = tmp_path / "data" / "splits" / "test.csv"
    rows, image_relative_path = _make_test_csv(split_path)
    image_path = dataset_root / image_relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), color=(25, 120, 50)).save(image_path)

    config_path = tmp_path / "config.yaml"
    _make_config(config_path, split_path)

    model = TinyClassifier().eval()
    checkpoint_path = tmp_path / "best_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    monkeypatch.setattr(
        phase9_experiment,
        "build_efficientnet_b0",
        lambda num_classes, pretrained: TinyClassifier(),
    )

    backend = None if failure_mode == "initialization" else FakeBackend(
        failure=backend_failure
    )

    def backend_factory() -> FakeBackend:
        if failure_mode == "initialization":
            raise ValueError("mock missing API key")
        return FakeBackend()

    output_dir = tmp_path / "output"
    result = phase9_experiment.run_single_image_experiment(
        config_path=config_path,
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        backend=backend,
        backend_factory=backend_factory,
        image_path=image_relative_path,
    )

    if backend is not None:
        assert len(backend.requests) == 1
        assert backend.requests[0].classifier.score_semantics == (
            "uncalibrated_softmax_score"
        )
    assert result["image"]["official_test_split_row_index"] == 0
    assert result["image"]["path"] == str(image_path.resolve())
    assert result["checkpoint"]["identifier"] == "best_checkpoint.pt"
    assert result["model"]["name"] == "EfficientNetB0"
    assert result["prediction"]["score_semantics"] == "uncalibrated_softmax_score"
    assert 0 <= result["prediction"]["softmax_score"] <= 1
    if backend is not None:
        assert result["prediction"]["softmax_score"] == pytest.approx(
            backend.requests[0].classifier.softmax_score
        )
    assert Path(result["gradcam_visualization_path"]).is_file()
    assert Path(result["result_json_path"]).is_file()
    saved = json.loads(Path(result["result_json_path"]).read_text(encoding="utf-8"))
    assert saved["status"] == (
        "completed" if failure_mode is None else "explanation_failed"
    )
    if failure_mode is not None:
        assert saved["validated_explanation"] is None
        assert saved["failure_status"]["stage"] == "structured_explanation"
        assert saved["failure_status"]["error_type"] == (
            "ValueError"
            if failure_mode == "initialization"
            else (
                "GeminiBackendError"
                if failure_mode == "safe_backend"
                else "RuntimeError"
            )
        )
        if failure_mode == "safe_backend":
            assert saved["failure_status"]["message"] == (
                "Gemini service error (HTTP 504)."
            )
        else:
            assert saved["failure_status"]["message"] == (
                "No validated VLM explanation was produced."
            )
            assert "secret test-token" not in json.dumps(saved)
    else:
        assert backend is not None
        explanation = saved["validated_explanation"]
        assert explanation["classifier"]["predicted_class"] == result["prediction"]["predicted_class"]
        assert explanation["classifier"]["class_index"] == result["prediction"]["class_index"]
        assert explanation["classifier"]["softmax_score"] == result["prediction"]["softmax_score"]
        assert explanation["classifier"]["softmax_score"] == pytest.approx(
            backend.requests[0].classifier.softmax_score
        )
        assert explanation["recommendations"] == {
            "status": "withheld_no_verified_source",
            "items": [],
        }
        assert "not a lesion segmentation mask" in explanation["limitations"][0].lower()
    assert "severity" in saved["scope_note"]
    assert "field reliability" in saved["scope_note"]

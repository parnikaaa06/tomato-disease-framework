
import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from PIL import Image

import evaluation.evaluate_gradcam as gradcam_evaluation
from evaluation.visualize_gradcam import load_checkpoint
from explainability.gradcam import generate_gradcam
from models.efficientnet_baseline import build_efficientnet_b0


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 3)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.features(image))
        return self.classifier(self.pool(features).flatten(1))


def test_gradcam_generates_finite_normalized_heatmap_and_restores_model_mode():
    model = TinyClassifier().train()
    image = torch.randn(1, 3, 24, 32)

    heatmap, class_idx = generate_gradcam(
        model=model,
        input_tensor=image,
        target_layer=model.features,
    )

    assert heatmap.shape == (24, 32)
    assert torch.isfinite(heatmap).all()
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1
    assert 0 <= class_idx < 3
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_gradcam_supports_phase3_efficientnet_b0_target_layer():
    model = build_efficientnet_b0(pretrained=False).eval()

    heatmap, class_idx = generate_gradcam(
        model,
        torch.randn(1, 3, 224, 224),
        model.features[-1],
    )

    assert heatmap.shape == (224, 224)
    assert torch.isfinite(heatmap).all()
    assert 0 <= class_idx < 10


@pytest.mark.parametrize(
    "image",
    [
        torch.randn(3, 24, 32),
        torch.randn(2, 3, 24, 32),
        torch.randn(1, 1, 24, 32),
        torch.randn(1, 3, 0, 32),
    ],
)
def test_gradcam_rejects_invalid_input_shapes(image: torch.Tensor):
    model = TinyClassifier()

    with pytest.raises(ValueError):
        generate_gradcam(model, image, model.features)


@pytest.mark.parametrize("class_idx", [-1, 3])
def test_gradcam_rejects_class_index_outside_model_range(class_idx: int):
    model = TinyClassifier().train()

    with pytest.raises(ValueError, match="class_idx"):
        generate_gradcam(
            model,
            torch.randn(1, 3, 24, 32),
            model.features,
            class_idx=class_idx,
        )

    assert model.training
    assert not model.features._forward_hooks


def test_gradcam_removes_hook_when_target_layer_is_not_used():
    model = TinyClassifier()
    target_layer = nn.Conv2d(3, 2, kernel_size=1)

    with pytest.raises(RuntimeError, match="capture"):
        generate_gradcam(
            model,
            torch.randn(1, 3, 24, 32),
            target_layer,
        )

    assert not target_layer._forward_hooks


def test_gradcam_computes_gradients_when_called_inside_no_grad():
    model = TinyClassifier()
    image = torch.randn(1, 3, 24, 32)

    with torch.no_grad():
        heatmap, _ = generate_gradcam(model, image, model.features)

    assert torch.isfinite(heatmap).all()


def test_load_checkpoint_restores_model_state_dict(tmp_path: Path):
    model = nn.Linear(3, 2)
    expected_state = {
        name: torch.full_like(value, fill_value=0.25)
        for name, value in model.state_dict().items()
    }
    checkpoint_path = tmp_path / "best_checkpoint.pt"
    torch.save({"model_state_dict": expected_state}, checkpoint_path)

    load_checkpoint(model, checkpoint_path, torch.device("cpu"))

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_state[name])


def test_load_checkpoint_reports_missing_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Baseline checkpoint"):
        load_checkpoint(nn.Linear(3, 2), tmp_path / "missing.pt", torch.device("cpu"))


def test_load_checkpoint_rejects_unexpected_checkpoint_format(tmp_path: Path):
    checkpoint_path = tmp_path / "invalid.pt"
    torch.save({"state_dict": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="model_state_dict"):
        load_checkpoint(nn.Linear(3, 2), checkpoint_path, torch.device("cpu"))


def make_split_rows(class_count: int = 10, images_per_class: int = 6):
    return [
        {"image_path": f"class_{class_index}/image_{image_index}.jpg",
         "class_name": f"class_{class_index}"}
        for class_index in range(class_count)
        for image_index in range(images_per_class)
    ]


def test_balanced_sampling_is_deterministic_and_without_replacement():
    rows = make_split_rows()

    sample_a = gradcam_evaluation.select_balanced_samples(
        rows, samples_per_class=3, seed=17
    )
    sample_b = gradcam_evaluation.select_balanced_samples(
        rows, samples_per_class=3, seed=17
    )

    assert sample_a == sample_b
    assert len(sample_a) == 30
    assert len({row["image_path"] for row in sample_a}) == 30
    assert {
        class_name: sum(row["class_name"] == class_name for row in sample_a)
        for class_name in {row["class_name"] for row in sample_a}
    } == {f"class_{index}": 3 for index in range(10)}


@pytest.mark.parametrize("samples_per_class", [0, -1])
def test_balanced_sampling_rejects_non_positive_sample_count(samples_per_class: int):
    with pytest.raises(ValueError, match="positive integer"):
        gradcam_evaluation.select_balanced_samples(
            make_split_rows(), samples_per_class=samples_per_class
        )


def test_balanced_sampling_reports_class_with_too_few_images():
    rows = make_split_rows(images_per_class=2)

    with pytest.raises(ValueError, match="cannot select 3"):
        gradcam_evaluation.select_balanced_samples(rows, samples_per_class=3)


def test_balanced_sampling_rejects_duplicate_image_paths():
    rows = [
        {"image_path": "same.jpg", "class_name": "class_a"},
        {"image_path": "same.jpg", "class_name": "class_a"},
    ]

    with pytest.raises(ValueError, match="duplicate image path"):
        gradcam_evaluation.select_balanced_samples(rows, samples_per_class=1)


def test_test_split_validation_requires_expected_size_and_classes():
    class_rows = make_split_rows(images_per_class=1)
    rows = [class_rows[index % len(class_rows)] for index in range(2725)]
    nine_class_rows = [
        class_rows[index % 9] for index in range(2725)
    ]

    with pytest.raises(ValueError, match="expected exactly 2725"):
        gradcam_evaluation.validate_test_split(rows[:-1], configured_class_count=10)

    with pytest.raises(ValueError, match="Configured class_count"):
        gradcam_evaluation.validate_test_split(rows, configured_class_count=9)

    with pytest.raises(ValueError, match="expected exactly 10"):
        gradcam_evaluation.validate_test_split(
            nine_class_rows, configured_class_count=10
        )


def test_evaluation_writes_per_image_visualizations_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset_root = tmp_path / "images"
    rows = [
        {"image_path": "class_a/a.jpg", "class_name": "class_a"},
        {"image_path": "class_b/b.jpg", "class_name": "class_b"},
    ]
    for row in rows:
        image_path = dataset_root / row["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color="green").save(image_path)

    gradcam_calls: list[int] = []

    def fake_generate_gradcam(
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
    ) -> tuple[torch.Tensor, int]:
        gradcam_calls.append(1)
        return torch.ones(224, 224), 0

    monkeypatch.setattr(
        gradcam_evaluation, "generate_gradcam", fake_generate_gradcam
    )
    model = TinyClassifier()
    output_dir = tmp_path / "output"
    records = gradcam_evaluation.evaluate_selected_samples(
        model=model,
        selected_rows=rows,
        dataset_root=dataset_root,
        class_names=["class_a", "class_b"],
        image_size=224,
        device=torch.device("cpu"),
        output_dir=output_dir,
    )
    csv_path, summary_path, summary = gradcam_evaluation.write_evaluation_outputs(
        records,
        output_dir,
        tmp_path / "best_checkpoint.pt",
        tmp_path / "test.csv",
    )

    assert len(gradcam_calls) == 2
    assert len(records) == 2
    assert records[0]["correctness"] is True
    assert records[1]["correctness"] is False
    assert all(Path(record["heatmap_path"]).is_file() for record in records)
    assert csv_path.is_file()
    assert summary_path.is_file()
    assert summary["sample_size"] == 2
    assert summary["per_class_counts"] == {"class_a": 1, "class_b": 1}
    assert summary["correct_predictions"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["accuracy_scope"] == (
        "balanced sample only; not the full test set"
    )
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    with summary_path.open("r", encoding="utf-8") as json_file:
        saved_summary = json.load(json_file)
    assert len(csv_rows) == 2
    assert set(csv_rows[0]) == set(gradcam_evaluation.RESULT_COLUMNS)
    assert csv_rows[0]["correctness"] == "True"
    assert saved_summary == summary


def test_evaluation_reports_missing_image_path(tmp_path: Path):
    rows = [{"image_path": "missing.jpg", "class_name": "class_a"}]

    with pytest.raises(FileNotFoundError, match="Image referenced by split is missing"):
        gradcam_evaluation.evaluate_selected_samples(
            model=TinyClassifier(),
            selected_rows=rows,
            dataset_root=tmp_path,
            class_names=["class_a"],
            image_size=224,
            device=torch.device("cpu"),
            output_dir=tmp_path / "out",
        )

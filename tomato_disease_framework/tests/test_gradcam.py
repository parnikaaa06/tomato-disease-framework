
from pathlib import Path

import pytest
import torch
from torch import nn

from evaluation.visualize_gradcam import load_checkpoint
from explainability.gradcam import generate_gradcam
from models.efficientnet_baseline import build_efficientnet_b0


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 4, kernel_size=3, padding=1)
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

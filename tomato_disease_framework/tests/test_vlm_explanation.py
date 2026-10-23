from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

from vlm.explanation import (
    MODEL_ID,
    ExplanationRequest,
    generate_structured_explanation,
)
from vlm.schema import (
    OUTPUT_SCHEMA,
    REQUIRED_LIMITATIONS,
    SCHEMA_VERSION,
    validate_output,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 3)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.features(image))
        return self.classifier(self.pool(features).flatten(1))


class FakeBackend:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = (
            valid_backend_response() if response is None else response
        )
        self.request: ExplanationRequest | None = None

    def explain(self, request: ExplanationRequest) -> dict[str, Any]:
        self.request = request
        return self.response


def valid_backend_response() -> dict[str, Any]:
    return {
        "observations": [
            {
                "text": "The image shows a green leaf against a plain background.",
                "evidence_sources": ["input_image"],
                "certainty": "direct_visual_observation",
            }
        ],
        "summary": (
            "The classifier predicts the supplied class; this output does not "
            "confirm a diagnosis."
        ),
        "claims": [
            {
                "text": "The heatmap highlights a broad region of the leaf.",
                "claim_type": "visual_observation",
                "evidence_sources": ["gradcam_heatmap"],
                "certainty": "supported",
            }
        ],
    }


def valid_output() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "classifier": {
            "model_id": MODEL_ID,
            "predicted_class": "Tomato___healthy",
            "class_index": 2,
            "softmax_score": 0.75,
            "score_semantics": "uncalibrated_softmax_score",
        },
        "visual_evidence": {
            "gradcam_target_class": "Tomato___healthy",
            "gradcam_target_layer": "model.features[-1]",
            "observations": valid_backend_response()["observations"],
        },
        "explanation": {
            "summary": "Cautious description.",
            "claims": valid_backend_response()["claims"],
        },
        "recommendations": {
            "status": "withheld_no_verified_source",
            "items": [],
        },
        "limitations": list(REQUIRED_LIMITATIONS),
    }


def test_schema_accepts_valid_versioned_explanation_and_withholds_advice():
    output = valid_output()

    validate_output(output)

    assert OUTPUT_SCHEMA["additionalProperties"] is False
    assert output["schema_version"] == "1.0"
    json.dumps(output, allow_nan=False)
    assert output["recommendations"] == {
        "status": "withheld_no_verified_source",
        "items": [],
    }


@pytest.mark.parametrize(
    "claim_text",
    [
        "Tomato___Late_blight is confirmed.",
        "The image confirms blight.",
        "This leaf definitely has a disease.",
        "This is definitely blight.",
        "The plant is infected with a pathogen.",
    ],
)
def test_schema_rejects_definitive_disease_confirmation_even_if_uncertain(
    claim_text: str,
):
    output = valid_output()
    output["explanation"]["claims"][0].update(
        text=claim_text,
        certainty="uncertain",
    )

    with pytest.raises(
        ValueError, match="must not claim definitive disease confirmation"
    ):
        validate_output(output)


@pytest.mark.parametrize(
    "cautious_text",
    [
        "The classifier predicts Tomato___Late_blight; this is not a diagnosis.",
        "The available image does not confirm a disease.",
        "There is no evidence that the image confirms an infection.",
        "No definitive evidence establishes an infection.",
    ],
)
def test_schema_allows_cautious_prediction_and_explicit_non_confirmation(
    cautious_text: str,
):
    output = valid_output()
    output["explanation"]["summary"] = cautious_text

    validate_output(output)


def test_backend_response_rejects_definitive_disease_claim_during_orchestration():
    response = valid_backend_response()
    response["claims"][0].update(
        text="The image proves that Tomato___Late_blight is present.",
        certainty="uncertain",
    )

    with pytest.raises(
        ValueError, match="must not claim definitive disease confirmation"
    ):
        generate_structured_explanation(
            TinyClassifier(),
            torch.randn(1, 3, 24, 24),
            Image.new("RGB", (24, 24)),
            ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"],
            FakeBackend(response),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda output: output["explanation"]["claims"][0].update(
                evidence_sources=[]
            ),
            "at least 1 items",
        ),
        (
            lambda output: output["explanation"]["claims"][0].update(text=""),
            "must not be empty",
        ),
        (
            lambda output: output["explanation"]["claims"][0].update(
                evidence_sources=["unsupported_source"]
            ),
            "must be one of",
        ),
        (
            lambda output: output["explanation"]["claims"][0].update(
                certainty="certain"
            ),
            "must be one of",
        ),
        (
            lambda output: output["recommendations"].update(
                status="recommendation_available"
            ),
            "must equal",
        ),
        (
            lambda output: output["recommendations"].update(items=["apply treatment"]),
            "at most 0 items",
        ),
        (
            lambda output: output.update(unapproved_field=True),
            "unsupported fields",
        ),
        (
            lambda output: output.update(schema_version="2.0"),
            "must equal",
        ),
        (
            lambda output: output["classifier"].update(softmax_score=float("nan")),
            "must be number",
        ),
        (
            lambda output: output["classifier"].update(class_index=10),
            "at most 9",
        ),
    ],
)
def test_schema_rejects_invalid_or_unsupported_claims(mutate, message: str):
    output = valid_output()
    mutate(output)

    with pytest.raises(ValueError, match=message):
        validate_output(output)


def test_schema_requires_gradcam_target_to_match_prediction():
    output = valid_output()
    output["visual_evidence"]["gradcam_target_class"] = "Tomato___Early_blight"

    with pytest.raises(ValueError, match="must match the classifier prediction"):
        validate_output(output)


def test_generation_preserves_classifier_output_and_does_not_change_weights():
    torch.manual_seed(31)
    model = TinyClassifier().train()
    input_tensor = torch.randn(1, 3, 24, 24)
    image = Image.new("RGB", (24, 24), color="green")
    backend = FakeBackend()
    weights_before = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    with torch.no_grad():
        logits = model(input_tensor)
        expected_index = int(logits.argmax(dim=1).item())
        expected_score = float(torch.softmax(logits, dim=1)[0, expected_index].item())

    output = generate_structured_explanation(
        model,
        input_tensor,
        image,
        ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"],
        backend,
    )

    assert backend.request is not None
    assert backend.request.classifier.predicted_class == output["classifier"]["predicted_class"]
    assert backend.request.classifier.class_index == expected_index
    assert output["classifier"]["model_id"] == MODEL_ID
    assert output["classifier"]["class_index"] == expected_index
    assert output["classifier"]["softmax_score"] == pytest.approx(expected_score)
    assert output["classifier"]["score_semantics"] == "uncalibrated_softmax_score"
    assert output["visual_evidence"]["gradcam_target_class"] == output["classifier"]["predicted_class"]
    assert backend.request.gradcam_heatmap.shape == (24, 24)
    assert backend.request.limitations == REQUIRED_LIMITATIONS
    assert output["recommendations"]["items"] == []
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights_before[key])


def test_backend_cannot_override_classifier_fields():
    response = valid_backend_response()
    response["classifier"] = {
        "predicted_class": "Tomato___Early_blight",
        "class_index": 0,
        "softmax_score": 1.0,
    }
    backend = FakeBackend(response)

    with pytest.raises(ValueError, match="exactly 'observations', 'summary', and 'claims'"):
        generate_structured_explanation(
            TinyClassifier(),
            torch.randn(1, 3, 24, 24),
            Image.new("RGB", (24, 24)),
            ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"],
            backend,
        )


def test_backend_response_with_missing_fields_is_rejected():
    backend = FakeBackend({})

    with pytest.raises(ValueError, match="exactly 'observations', 'summary', and 'claims'"):
        generate_structured_explanation(
            TinyClassifier(),
            torch.randn(1, 3, 24, 24),
            Image.new("RGB", (24, 24)),
            ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"],
            backend,
        )


def test_backend_error_propagates_and_model_mode_is_restored():
    class FailingBackend:
        def explain(self, _request: ExplanationRequest) -> dict[str, Any]:
            raise RuntimeError("deterministic backend failure")

    model = TinyClassifier().train()
    with pytest.raises(RuntimeError, match="deterministic backend failure"):
        generate_structured_explanation(
            model,
            torch.randn(1, 3, 24, 24),
            Image.new("RGB", (24, 24)),
            ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"],
            FailingBackend(),
        )

    assert model.training


def test_missing_backend_fails_explicitly():
    with pytest.raises(TypeError, match="backend must implement explain"):
        generate_structured_explanation(
            TinyClassifier(),
            torch.randn(1, 3, 24, 24),
            Image.new("RGB", (24, 24)),
            ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"],
            None,  # type: ignore[arg-type]
        )


def test_frozen_classifier_result_cannot_be_mutated_by_backend():
    class MutatingBackend:
        def explain(self, request: ExplanationRequest) -> dict[str, Any]:
            request.classifier.predicted_class = "changed"  # type: ignore[misc]
            return valid_backend_response()

    with pytest.raises(FrozenInstanceError):
        generate_structured_explanation(
            TinyClassifier(),
            torch.randn(1, 3, 24, 24),
            Image.new("RGB", (24, 24)),
            ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"],
            MutatingBackend(),
        )


def test_non_finite_classifier_logits_fail_without_backend_invocation():
    class NonFiniteClassifier(TinyClassifier):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            logits = super().forward(image)
            logits[0, 0] = float("nan")
            return logits

    backend = FakeBackend()
    with pytest.raises(ValueError, match="finite values"):
        generate_structured_explanation(
            NonFiniteClassifier(),
            torch.randn(1, 3, 24, 24),
            Image.new("RGB", (24, 24)),
            ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"],
            backend,
        )
    assert backend.request is None

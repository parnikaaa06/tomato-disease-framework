from __future__ import annotations

import importlib
import json
import socket
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

from vlm.explanation import (
    ExplanationRequest,
    generate_structured_explanation,
)
from vlm.gemini_backend import (
    DEFAULT_GEMINI_MODEL,
    GeminiBackend,
    GeminiBackendError,
)
from vlm.schema import BACKEND_RESPONSE_SCHEMA, REQUIRED_LIMITATIONS


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 3)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.features(image))
        return self.classifier(self.pool(features).flatten(1))


def backend_response() -> dict[str, Any]:
    return {
        "observations": [
            {
                "text": "A green leaf is visible.",
                "evidence_sources": ["input_image"],
                "certainty": "direct_visual_observation",
            }
        ],
        "summary": "The model output is a prediction, not a confirmed diagnosis.",
        "claims": [
            {
                "text": "The heatmap shows a broad high-attribution region.",
                "claim_type": "visual_observation",
                "evidence_sources": ["gradcam_heatmap"],
                "certainty": "supported",
            }
        ],
    }


def make_request() -> ExplanationRequest:
    from vlm.explanation import ClassifierResult
    from vlm.prompts import EXPLANATION_PROMPT

    return ExplanationRequest(
        prompt=EXPLANATION_PROMPT,
        image=Image.new("RGBA", (3, 2), color=(10, 20, 30, 255)),
        gradcam_heatmap=torch.tensor([[0.0, 0.5], [1.0, 0.25]]),
        classifier=ClassifierResult(
            model_id="efficientnet_b0_phase3_baseline",
            predicted_class="Tomato___healthy",
            class_index=2,
            softmax_score=0.8,
        ),
        limitations=REQUIRED_LIMITATIONS,
    )


class FakeModels:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        next_response = self.responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response


def fake_client(responses: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(models=FakeModels(responses))


def text_response(body: Any) -> SimpleNamespace:
    text = body if isinstance(body, str) else json.dumps(body)
    return SimpleNamespace(text=text, candidates=[])


def test_gemini_backend_sends_labeled_images_and_structured_schema():
    client = fake_client([text_response(backend_response())])
    backend = GeminiBackend(client=client, model="test-vision-model")
    request = make_request()

    result = backend.explain(request)

    call = client.models.calls[0]
    contents = call["contents"]
    assert call["model"] == "test-vision-model"
    assert call["config"]["response_mime_type"] == "application/json"
    assert call["config"]["response_schema"] == BACKEND_RESPONSE_SCHEMA
    assert call["config"]["temperature"] == 0
    assert "Tomato___healthy" in contents[0]
    assert "uncalibrated_softmax_score" in contents[0]
    assert "not a lesion mask" in contents[0]
    assert isinstance(contents[1], Image.Image)
    assert contents[1].mode == "RGB"
    assert contents[1].getpixel((0, 0)) == (10, 20, 30)
    assert "Grad-CAM heatmap" in contents[2]
    assert isinstance(contents[3], Image.Image)
    assert contents[3].mode == "L"
    assert contents[3].getpixel((0, 0)) == 0
    assert contents[3].getpixel((1, 0)) == 128
    assert result == backend_response()


def test_missing_api_key_fails_without_constructing_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
        GeminiBackend()


@pytest.mark.parametrize("model", ["", "model with spaces", "bad\nmodel"])
def test_invalid_model_configuration_is_rejected(model: str):
    with pytest.raises(ValueError, match="GEMINI_MODEL"):
        GeminiBackend(client=fake_client([]), model=model)


def test_default_model_and_environment_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    default_backend = GeminiBackend(client=fake_client([]))
    assert DEFAULT_GEMINI_MODEL == "gemini-3.6-flash"
    assert default_backend.model == "gemini-3.6-flash"
    assert default_backend.timeout_ms == 20_000

    monkeypatch.setenv("GEMINI_MODEL", "gemini-configured-model")
    assert GeminiBackend(client=fake_client([])).model == "gemini-configured-model"


@pytest.mark.parametrize("timeout", ["0", "-1", "60001", "not-a-number"])
def test_invalid_timeout_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    timeout: str,
):
    monkeypatch.setenv("GEMINI_TIMEOUT_MS", timeout)

    with pytest.raises(ValueError, match="GEMINI_TIMEOUT_MS"):
        GeminiBackend(client=fake_client([]))


def test_sdk_client_receives_configured_timeout_without_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    class FakeHttpRetryOptions:
        def __init__(self, *, attempts: int) -> None:
            self.attempts = attempts

    class FakeHttpOptions:
        def __init__(self, *, timeout: int, retry_options: Any) -> None:
            self.timeout = timeout
            self.retry_options = retry_options

    class FakeSdkClient:
        def __init__(self, *, api_key: str, http_options: Any) -> None:
            captured["api_key"] = api_key
            captured["http_options"] = http_options
            self.models = FakeModels([])

    google_module = ModuleType("google")
    google_module.__path__ = []
    genai_module = ModuleType("google.genai")
    types_module = ModuleType("google.genai.types")
    genai_module.Client = FakeSdkClient
    genai_module.types = types_module
    types_module.HttpOptions = FakeHttpOptions
    types_module.HttpRetryOptions = FakeHttpRetryOptions
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-secret")
    monkeypatch.setenv("GEMINI_TIMEOUT_MS", "12345")

    backend = GeminiBackend()

    assert backend.timeout_ms == 12_345
    assert captured["http_options"].timeout == 12_345
    assert captured["http_options"].retry_options.attempts == 1


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(text=None, candidates=[]),
        SimpleNamespace(text="", candidates=[]),
        SimpleNamespace(text="not JSON", candidates=[]),
        text_response({"observations": [], "summary": "", "claims": []}),
        SimpleNamespace(
            text=None,
            candidates=[SimpleNamespace(finish_reason="SAFETY")],
        ),
    ],
)
def test_missing_malformed_refused_or_invalid_response_fails(response: Any):
    backend = GeminiBackend(client=fake_client([response]), sleep_fn=lambda _: None)

    with pytest.raises(GeminiBackendError):
        backend.explain(make_request())


def test_response_text_accessor_failure_is_reported_safely():
    class BrokenResponse:
        candidates: list[Any] = []

        @property
        def text(self) -> str:
            raise RuntimeError("provider response internals")

    backend = GeminiBackend(client=fake_client([BrokenResponse()]))

    with pytest.raises(
        GeminiBackendError,
        match="empty, malformed, or refused response",
    ):
        backend.explain(make_request())


class MockApiError(Exception):
    def __init__(self, code: int, message: str = "mock error") -> None:
        super().__init__(message)
        self.code = code


@pytest.mark.parametrize(
    ("code", "expected_text"),
    [
        (401, "authentication failed"),
        (403, "permission denied"),
        (429, "rate limit or quota"),
        (500, "service error"),
        (503, "service error"),
        (504, "service error"),
    ],
)
def test_api_errors_are_reported_safely_and_retries_are_bounded(
    code: int,
    expected_text: str,
):
    sleeps: list[float] = []
    key_material = "must-not-appear-in-errors"
    client = fake_client(
        [
            MockApiError(code, f"mock error includes {key_material}"),
            MockApiError(code, f"second error includes {key_material}"),
            text_response(backend_response()),
        ]
    )
    backend = GeminiBackend(
        client=client,
        max_attempts=2,
        sleep_fn=sleeps.append,
    )

    with pytest.raises(GeminiBackendError, match=expected_text) as error:
        backend.explain(make_request())

    assert len(client.models.calls) == (1 if code in {401, 403} else 2)
    assert sleeps == ([0.5] if code in {429, 500, 503, 504} else [])
    assert key_material not in str(error.value)


@pytest.mark.parametrize("code", [400, 501])
def test_non_transient_http_errors_are_not_retried(code: int):
    client = fake_client([MockApiError(code), text_response(backend_response())])
    backend = GeminiBackend(client=client)

    with pytest.raises(GeminiBackendError):
        backend.explain(make_request())

    assert len(client.models.calls) == 1


def test_transient_timeout_retries_once_then_succeeds():
    client = fake_client(
        [TimeoutError("timeout"), text_response(backend_response())]
    )
    sleeps: list[float] = []
    backend = GeminiBackend(client=client, sleep_fn=sleeps.append)

    assert backend.explain(make_request()) == backend_response()
    assert len(client.models.calls) == 2
    assert sleeps == [0.5]


def test_exhausted_timeout_has_a_clear_error_without_provider_details():
    client = fake_client([TimeoutError("secret-bearing provider detail")])
    backend = GeminiBackend(client=client, max_attempts=1)

    with pytest.raises(GeminiBackendError, match="network request timed out") as error:
        backend.explain(make_request())

    assert "secret-bearing provider detail" not in str(error.value)


def test_invalid_heatmap_fails_before_api_request():
    client = fake_client([])
    request = make_request()
    request = ExplanationRequest(
        prompt=request.prompt,
        image=request.image,
        gradcam_heatmap=torch.tensor([[float("nan")]]),
        classifier=request.classifier,
        limitations=request.limitations,
    )

    with pytest.raises(ValueError, match="finite values"):
        GeminiBackend(client=client).explain(request)
    assert client.models.calls == []


def test_gemini_backend_preserves_classifier_and_withholds_recommendations():
    client = fake_client([text_response(backend_response())])
    backend = GeminiBackend(client=client)
    model = TinyClassifier().eval()
    weights_before = {
        key: tensor.detach().clone() for key, tensor in model.state_dict().items()
    }
    input_tensor = torch.randn(1, 3, 24, 24)
    with torch.no_grad():
        logits = model(input_tensor)
        expected_index = int(logits.argmax(dim=1).item())
        expected_score = float(
            torch.softmax(logits, dim=1)[0, expected_index].item()
        )

    output = generate_structured_explanation(
        model,
        input_tensor,
        Image.new("RGB", (24, 24)),
        ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"],
        backend,
    )

    assert output["classifier"]["class_index"] == expected_index
    assert output["classifier"]["predicted_class"] == [
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___healthy",
    ][expected_index]
    assert output["classifier"]["softmax_score"] == pytest.approx(expected_score)
    assert output["visual_evidence"]["gradcam_target_class"] == output["classifier"]["predicted_class"]
    assert output["limitations"] == list(REQUIRED_LIMITATIONS)
    assert output["recommendations"] == {
        "status": "withheld_no_verified_source",
        "items": [],
    }
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, weights_before[key])


def test_import_does_not_import_sdk_or_open_network(
    monkeypatch: pytest.MonkeyPatch,
):
    original_import = __import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "google" or name.startswith("google.genai"):
            raise AssertionError("Gemini SDK must be imported lazily.")
        return original_import(name, *args, **kwargs)

    def forbidden_connect(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Module import must not access the network.")

    monkeypatch.setattr("builtins.__import__", guarded_import)
    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    importlib.reload(sys.modules["vlm.gemini_backend"])

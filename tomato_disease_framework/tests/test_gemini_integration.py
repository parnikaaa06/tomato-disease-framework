from __future__ import annotations

import os

import pytest
import torch
from PIL import Image

from vlm.explanation import ClassifierResult, ExplanationRequest
from vlm.gemini_backend import GeminiBackend
from vlm.prompts import EXPLANATION_PROMPT
from vlm.schema import REQUIRED_LIMITATIONS


@pytest.mark.gemini_integration
@pytest.mark.skipif(
    os.environ.get("RUN_GEMINI_INTEGRATION") != "1",
    reason="Set RUN_GEMINI_INTEGRATION=1 to make one real Gemini API request.",
)
def test_gemini_vision_structured_output_smoke() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.fail("GEMINI_API_KEY must be configured for the opt-in smoke test.")

    request = ExplanationRequest(
        prompt=EXPLANATION_PROMPT,
        image=Image.new("RGB", (16, 16), color=(50, 120, 40)),
        gradcam_heatmap=torch.zeros((4, 4)),
        classifier=ClassifierResult(
            model_id="efficientnet_b0_phase3_baseline",
            predicted_class="Tomato___healthy",
            class_index=9,
            softmax_score=0.75,
        ),
        limitations=REQUIRED_LIMITATIONS,
    )
    backend = GeminiBackend()
    try:
        response = backend.explain(request)
    finally:
        backend.close()

    assert set(response) == {"observations", "summary", "claims"}
    assert response["summary"]

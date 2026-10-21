"""Provider-neutral structured explanations for classifier and Grad-CAM evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

import torch
from PIL import Image
from torch import nn

from explainability.gradcam import generate_gradcam
from vlm.prompts import EXPLANATION_PROMPT
from vlm.schema import REQUIRED_LIMITATIONS, validate_output


MODEL_ID = "efficientnet_b0_phase3_baseline"
GRADCAM_TARGET_LAYER = "model.features[-1]"
BACKEND_RESPONSE_FIELDS = frozenset({"observations", "summary", "claims"})


@dataclass(frozen=True)
class ClassifierResult:
    """Immutable classifier output supplied to the explanation backend."""

    model_id: str
    predicted_class: str
    class_index: int
    softmax_score: float
    score_semantics: str = "uncalibrated_softmax_score"


@dataclass(frozen=True)
class ExplanationRequest:
    """Read-only-by-contract evidence passed to an injected VLM backend."""

    prompt: str
    image: Image.Image
    gradcam_heatmap: torch.Tensor
    classifier: ClassifierResult
    limitations: tuple[str, ...]


class ExplanationBackend(Protocol):
    """Interface implemented by local, remote, or deterministic test backends."""

    def explain(self, request: ExplanationRequest) -> Mapping[str, Any]:
        """Return only observations, summary, and evidence-linked claims."""
        ...


def generate_structured_explanation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    image: Image.Image,
    class_names: list[str] | tuple[str, ...],
    backend: ExplanationBackend,
) -> dict[str, Any]:
    """Build validated JSON while preserving the classifier's prediction.

    This function makes no provider/network calls. A backend must be injected
    explicitly and can author explanation content only; classifier output,
    limitations, and the withheld-recommendations record are application-owned.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.")
    if not isinstance(class_names, (list, tuple)) or not class_names:
        raise ValueError("class_names must be a non-empty list or tuple.")
    if any(not isinstance(name, str) or not name for name in class_names):
        raise ValueError("class_names must contain non-empty class labels.")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be unique.")
    if not callable(getattr(backend, "explain", None)):
        raise TypeError("backend must implement explain(request).")

    features = getattr(model, "features", None)
    if features is None:
        raise ValueError("Model must expose EfficientNet features for Grad-CAM.")
    try:
        target_layer = features[-1]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError("Model features must contain a final target layer.") from error

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(input_tensor)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Model output must be a tensor of class logits.")
        if (
            logits.ndim != 2
            or logits.shape[0] != 1
            or logits.shape[1] != len(class_names)
            or logits.shape[1] == 0
        ):
            raise ValueError(
                "Model output must have shape [1, len(class_names)]."
            )
        if not torch.isfinite(logits).all():
            raise ValueError("Model logits must contain only finite values.")

        class_index = int(logits.argmax(dim=1).item())
        probabilities = torch.softmax(logits, dim=1)
        softmax_score = float(probabilities[0, class_index].item())
        if not 0.0 <= softmax_score <= 1.0:
            raise ValueError("Classifier softmax score must be within [0, 1].")
        classifier = ClassifierResult(
            model_id=MODEL_ID,
            predicted_class=class_names[class_index],
            class_index=class_index,
            softmax_score=softmax_score,
        )

        heatmap, gradcam_class_index = generate_gradcam(
            model=model,
            input_tensor=input_tensor,
            target_layer=target_layer,
            class_idx=class_index,
        )
        if gradcam_class_index != class_index:
            raise RuntimeError(
                "Grad-CAM target class differs from the classifier prediction."
            )
        if heatmap.ndim != 2 or not torch.isfinite(heatmap).all():
            raise RuntimeError("Grad-CAM returned an invalid heatmap.")

        request = ExplanationRequest(
            prompt=EXPLANATION_PROMPT,
            image=image.copy(),
            gradcam_heatmap=heatmap.detach().cpu().clone(),
            classifier=classifier,
            limitations=REQUIRED_LIMITATIONS,
        )
        backend_response = backend.explain(request)
        if not isinstance(backend_response, Mapping):
            raise TypeError("Backend response must be a JSON object mapping.")
        response_fields = set(backend_response)
        if response_fields != BACKEND_RESPONSE_FIELDS:
            raise ValueError(
                "Backend response must contain exactly "
                "'observations', 'summary', and 'claims'."
            )

        classifier_output = asdict(classifier)
        output: dict[str, Any] = {
            "schema_version": "1.0",
            "classifier": classifier_output,
            "visual_evidence": {
                "gradcam_target_class": classifier.predicted_class,
                "gradcam_target_layer": GRADCAM_TARGET_LAYER,
                "observations": backend_response["observations"],
            },
            "explanation": {
                "summary": backend_response["summary"],
                "claims": backend_response["claims"],
            },
            "recommendations": {
                "status": "withheld_no_verified_source",
                "items": [],
            },
            "limitations": list(REQUIRED_LIMITATIONS),
        }
        validate_output(
            output,
            expected_class=classifier.predicted_class,
            expected_class_index=classifier.class_index,
        )
        if output["classifier"] != classifier_output:
            raise RuntimeError("Classifier output changed during explanation generation.")
        return output
    finally:
        model.train(was_training)

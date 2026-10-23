"""Google Gemini vision backend for the provider-neutral Phase 9 contract."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image

from vlm.explanation import ExplanationRequest
from vlm.schema import BACKEND_RESPONSE_SCHEMA, validate_backend_response


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_TIMEOUT_MS = 20_000
MAX_TIMEOUT_MS = 60_000
DEFAULT_MAX_ATTEMPTS = 2
_VALID_MODEL_NAME = re.compile(r"^[A-Za-z0-9._/-]+$")
_SAFE_VALIDATION_DIAGNOSTIC = re.compile(
    r"^(?P<path>\$[A-Za-z0-9_.\[\]]+)\s+"
    r"(?P<reason>"
    r"must not claim definitive disease confirmation from image or classifier evidence"
    r"|must be (?:object|array|string|integer|number)"
    r"|must not be empty"
    r"|must contain at least \d+ items"
    r"|must be at least \d+(?:\.\d+)?"
    r"|must be at most \d+(?:\.\d+)?"
    r"|must be one of"
    r"|must equal"
    r"|has unsupported fields"
    r"|is missing required fields"
    r")"
)
_RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class GeminiBackendError(RuntimeError):
    """Safe, user-facing error for Gemini configuration and response failures."""


def _heatmap_image(heatmap: torch.Tensor) -> Image.Image:
    """Convert the Grad-CAM tensor into a labeled grayscale image input."""
    if not isinstance(heatmap, torch.Tensor):
        raise TypeError("Grad-CAM evidence must be a torch.Tensor.")
    if heatmap.ndim != 2 or heatmap.numel() == 0:
        raise ValueError("Grad-CAM evidence must be a non-empty 2D tensor.")

    values = heatmap.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("Grad-CAM evidence must contain only finite values.")
    if values.min().item() < 0 or values.max().item() > 1:
        raise ValueError("Grad-CAM evidence values must be in the range [0, 1].")

    pixels = np.rint(values.numpy() * 255).astype(np.uint8)
    return Image.fromarray(pixels)


def _response_was_refused(response: Any) -> bool:
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None)
    if block_reason is not None and "UNSPECIFIED" not in str(block_reason).upper():
        if str(block_reason).strip():
            return True

    for candidate in getattr(response, "candidates", None) or ():
        finish_reason = getattr(candidate, "finish_reason", None)
        reason = getattr(finish_reason, "name", str(finish_reason)).upper()
        if any(token in reason for token in ("SAFETY", "BLOCK", "RECITATION", "PROHIBITED")):
            return True
    return False


def _status_code(error: Exception) -> int | None:
    for attribute in ("status_code", "code", "status"):
        value = getattr(error, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            match = re.search(r"\b([1-5][0-9]{2})\b", value)
            if match:
                return int(match.group(1))
    return None


def _is_transient(error: Exception, status_code: int | None) -> bool:
    if status_code in _RETRYABLE_HTTP_STATUS_CODES:
        return True
    if status_code is not None:
        return False
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return True
    error_name = type(error).__name__.lower()
    return any(word in error_name for word in ("timeout", "connect", "network", "transport"))


def _safe_api_error_message(status_code: int | None) -> str:
    if status_code == 401:
        return "Gemini authentication failed (HTTP 401); verify GEMINI_API_KEY."
    if status_code == 403:
        return "Gemini permission denied (HTTP 403); check API key permissions."
    if status_code == 402:
        return "Gemini billing or prepaid quota is unavailable (HTTP 402)."
    if status_code == 429:
        return "Gemini rate limit or quota exceeded (HTTP 429)."
    if status_code == 408:
        return "Gemini request timed out (HTTP 408)."
    if status_code is not None and status_code >= 500:
        return f"Gemini service error (HTTP {status_code})."
    if status_code is not None:
        return f"Gemini request failed (HTTP {status_code})."
    return "Gemini request failed because of a network or SDK error."


def _safe_request_error_message(
    error: Exception,
    status_code: int | None,
) -> str:
    if status_code is not None:
        return _safe_api_error_message(status_code)
    error_name = type(error).__name__.lower()
    if "timeout" in error_name:
        return "Gemini network request timed out."
    if any(word in error_name for word in ("connect", "network", "transport")):
        return "Gemini network connection failed."
    return _safe_api_error_message(None)


def _safe_schema_error_message(error: Exception) -> str:
    """Retain schema path and rule without exposing response values."""
    match = _SAFE_VALIDATION_DIAGNOSTIC.match(str(error))
    if match:
        return f"{match.group('path')}: {match.group('reason')}."
    return f"local schema validation failed ({type(error).__name__})."


class GeminiBackend:
    """Call Gemini with two images and the existing strict backend response schema.

    The SDK is imported and the client constructed only when this backend is
    explicitly initialized without an injected client. Network access occurs
    only in ``explain``. Transient failures are attempted at most twice by
    default, with a short exponential delay; authentication, permission, and
    malformed-request errors are never retried. The SDK itself is configured
    for one attempt, so only this class applies retries.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        timeout_ms: int | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        configured_model = model if model is not None else os.environ.get(
            "GEMINI_MODEL", DEFAULT_GEMINI_MODEL
        )
        if (
            not isinstance(configured_model, str)
            or not configured_model
            or not _VALID_MODEL_NAME.fullmatch(configured_model)
        ):
            raise ValueError("GEMINI_MODEL must be a non-empty model identifier.")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise ValueError("max_attempts must be an integer from 1 to 3.")
        if not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be an integer from 1 to 3.")
        timeout_source = "timeout_ms"
        if timeout_ms is None:
            configured_timeout = os.environ.get("GEMINI_TIMEOUT_MS")
            if configured_timeout is None:
                timeout_ms = DEFAULT_TIMEOUT_MS
            else:
                timeout_source = "GEMINI_TIMEOUT_MS"
                try:
                    timeout_ms = int(configured_timeout)
                except ValueError:
                    raise ValueError(
                        f"{timeout_source} must be an integer from 1 to "
                        f"{MAX_TIMEOUT_MS}."
                    ) from None
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or not 1 <= timeout_ms <= MAX_TIMEOUT_MS
        ):
            raise ValueError(
                f"{timeout_source} must be an integer from 1 to "
                f"{MAX_TIMEOUT_MS}."
            )

        self.model = configured_model
        self.max_attempts = max_attempts
        self.timeout_ms = timeout_ms
        self.sleep_fn = sleep_fn
        self._owns_client = client is None

        if client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key or not api_key.strip():
                raise ValueError(
                    "GEMINI_API_KEY is required to initialize GeminiBackend."
                )
            try:
                from google import genai
                from google.genai import types
            except ImportError as error:
                raise RuntimeError(
                    "google-genai is required; install project dependencies with "
                    "`python -m pip install -r requirements.txt`."
                ) from error
            try:
                client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(
                        timeout=self.timeout_ms,
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                )
            except Exception:
                raise GeminiBackendError(
                    "Gemini SDK client initialization failed; verify configuration."
                ) from None
            finally:
                del api_key
        self._client = client

    def _request_contents(self, request: ExplanationRequest) -> list[Any]:
        if not isinstance(request, ExplanationRequest):
            raise TypeError("request must be an ExplanationRequest.")
        if not isinstance(request.image, Image.Image):
            raise TypeError("request.image must be a PIL Image.")
        if not isinstance(request.classifier.predicted_class, str):
            raise ValueError("Classifier context is invalid.")

        original = request.image.copy().convert("RGB")
        heatmap = _heatmap_image(request.gradcam_heatmap)
        classifier_context = {
            "model_id": request.classifier.model_id,
            "predicted_class": request.classifier.predicted_class,
            "class_index": request.classifier.class_index,
            "softmax_score": request.classifier.softmax_score,
            "score_semantics": request.classifier.score_semantics,
        }
        context_text = (
            f"{request.prompt}\n\n"
            "Fixed classifier context (report as a model prediction, not a "
            "visually confirmed diagnosis):\n"
            f"{json.dumps(classifier_context, ensure_ascii=True, allow_nan=False)}\n"
            "Application limitations:\n"
            + "\n".join(f"- {item}" for item in request.limitations)
            + "\n\nThe first image is the original input leaf image. "
            "The second image is a grayscale Grad-CAM attribution heatmap; "
            "brighter values indicate greater relative attribution for the "
            "predicted class. It is not a lesion mask."
        )
        return [
            context_text,
            original,
            "Grad-CAM heatmap image (not a lesion segmentation mask):",
            heatmap,
        ]

    def explain(self, request: ExplanationRequest) -> dict[str, Any]:
        """Return only validated observations, summary, and evidence-linked claims."""
        contents = self._request_contents(request)
        response: Any = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": BACKEND_RESPONSE_SCHEMA,
                        "temperature": 0,
                    },
                )
                break
            except Exception as error:
                status_code = _status_code(error)
                transient = _is_transient(error, status_code)
                if transient and attempt < self.max_attempts:
                    self.sleep_fn(0.5 * (2 ** (attempt - 1)))
                    continue
                raise GeminiBackendError(
                    _safe_request_error_message(error, status_code)
                ) from None

        if response is None or _response_was_refused(response):
            raise GeminiBackendError("Gemini refused the request or returned no response.")
        try:
            response_text = getattr(response, "text", None)
        except Exception:
            raise GeminiBackendError(
                "Gemini returned an empty, malformed, or refused response."
            ) from None
        if not isinstance(response_text, str) or not response_text.strip():
            raise GeminiBackendError("Gemini returned an empty or refused response.")
        try:
            parsed_response = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            raise GeminiBackendError("Gemini returned malformed JSON.") from None
        try:
            validate_backend_response(parsed_response)
        except (TypeError, ValueError) as error:
            raise GeminiBackendError(
                "Gemini response violates the local explanation schema: "
                f"{_safe_schema_error_message(error)}"
            ) from None
        return parsed_response

    def close(self) -> None:
        """Close a client created by this backend; injected clients remain caller-owned."""
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

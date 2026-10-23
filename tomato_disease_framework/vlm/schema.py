"""Strict, dependency-free validation for Phase 9 structured explanations."""

from __future__ import annotations

import math
import re
from typing import Any


SCHEMA_VERSION = "1.0"
REQUIRED_LIMITATIONS = (
    "Grad-CAM is not a lesion segmentation mask.",
    "The classifier softmax score is uncalibrated.",
    "Field reliability has not been established.",
)

OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Phase 9 Structured Tomato Disease Explanation",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "classifier",
        "visual_evidence",
        "explanation",
        "recommendations",
        "limitations",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": SCHEMA_VERSION},
        "classifier": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "model_id",
                "predicted_class",
                "class_index",
                "softmax_score",
                "score_semantics",
            ],
            "properties": {
                "model_id": {"type": "string", "minLength": 1},
                "predicted_class": {"type": "string", "minLength": 1},
                "class_index": {"type": "integer", "minimum": 0, "maximum": 9},
                "softmax_score": {"type": "number", "minimum": 0, "maximum": 1},
                "score_semantics": {
                    "type": "string",
                    "const": "uncalibrated_softmax_score",
                },
            },
        },
        "visual_evidence": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "gradcam_target_class",
                "gradcam_target_layer",
                "observations",
            ],
            "properties": {
                "gradcam_target_class": {"type": "string", "minLength": 1},
                "gradcam_target_layer": {
                    "type": "string",
                    "const": "model.features[-1]",
                },
                "observations": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["text", "evidence_sources", "certainty"],
                        "properties": {
                            "text": {"type": "string", "minLength": 1},
                            "evidence_sources": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "string",
                                    "enum": ["input_image", "gradcam_heatmap"],
                                },
                            },
                            "certainty": {
                                "type": "string",
                                "enum": [
                                    "direct_visual_observation",
                                    "uncertain",
                                ],
                            },
                        },
                    },
                },
            },
        },
        "explanation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary", "claims"],
            "properties": {
                "summary": {"type": "string", "minLength": 1},
                "claims": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "text",
                            "claim_type",
                            "evidence_sources",
                            "certainty",
                        ],
                        "properties": {
                            "text": {"type": "string", "minLength": 1},
                            "claim_type": {
                                "type": "string",
                                "enum": [
                                    "visual_observation",
                                    "interpretation",
                                ],
                            },
                            "evidence_sources": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "string",
                                    "enum": ["input_image", "gradcam_heatmap"],
                                },
                            },
                            "certainty": {
                                "type": "string",
                                "enum": ["supported", "uncertain"],
                            },
                        },
                    },
                },
            },
        },
        "recommendations": {
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "items"],
            "properties": {
                "status": {
                    "type": "string",
                    "const": "withheld_no_verified_source",
                },
                "items": {"type": "array", "maxItems": 0},
            },
        },
        "limitations": {
            "type": "array",
            "minItems": 3,
            "items": {"type": "string", "minLength": 1},
        },
    },
}

def _gemini_compatible_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Retain Gemini structured-output JSON Schema keywords from the contract."""
    supported_keywords = {"type", "required", "properties", "items", "enum"}
    compatible = {
        key: value
        for key, value in schema.items()
        if key in supported_keywords and key not in {"properties", "items"}
    }
    if "properties" in schema:
        compatible["properties"] = {
            key: _gemini_compatible_schema(value)
            for key, value in schema["properties"].items()
        }
    if "items" in schema:
        compatible["items"] = _gemini_compatible_schema(schema["items"])
    return compatible


_BACKEND_RESPONSE_CONTRACT: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["observations", "summary", "claims"],
    "properties": {
        "observations": OUTPUT_SCHEMA["properties"]["visual_evidence"]["properties"][
            "observations"
        ],
        "summary": OUTPUT_SCHEMA["properties"]["explanation"]["properties"][
            "summary"
        ],
        "claims": OUTPUT_SCHEMA["properties"]["explanation"]["properties"]["claims"],
    },
}

BACKEND_RESPONSE_SCHEMA: dict[str, Any] = _gemini_compatible_schema(
    _BACKEND_RESPONSE_CONTRACT
)

_DISEASE_REFERENCE = (
    r"(?:disease|infection|infected|diseased|pathogen|blight|mosaic|mildew|"
    r"mold|rot|canker|wilt|virus|Tomato___[A-Za-z0-9_]+)"
)
_NEGATED_CONFIRMATION_PATTERNS = (
    re.compile(
        r"\bnot\s+(?:a\s+)?(?:definitively\s+)?confirmed\s+diagnos(?:is|tic)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:does|do|did|is|are|was|were|has|have)\s+not\s+"
        r"(?:definitively\s+)?(?:confirm|prove|diagnos\w*|show|have)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:cannot|can't|never)\s+"
        r"(?:definitively\s+)?(?:confirm|prove|diagnos\w*|show|have)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_DISEASE_REFERENCE}\b.{{0,30}}\b(?:is|was|has been)\s+not\s+"
        r"(?:definitively\s+)?(?:confirmed|proven|diagnosed|established)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:is|are|was|were)\s+not\s+(?:definitively\s+)?"
        rf"(?:a|an|the)?\s*{_DISEASE_REFERENCE}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdiagnos\w*\b.{0,35}\b(?:is|was|has been)\s+not\s+"
        r"(?:definitively\s+)?(?:confirmed|proven|established)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bnot\s+(?:definitively\s+)?diagnos\w*\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:definitive\s+)?evidence\b[^,.!?]{0,45}"
        r"\b(?:confirm\w*|prove\w*|diagnos\w*|confirmed|proven|diagnosed)\b",
        re.IGNORECASE,
    ),
)
_DEFINITIVE_DISEASE_PATTERNS = (
    re.compile(
        r"\b(?:confirm\w*|prove\w*|establish\w*)\b.{0,60}"
        r"\b(?:the\s+)?diagnos(?:is|tic confirmation)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdiagnos(?:is|tic confirmation)\b.{0,35}"
        r"(?:(?:is|was|has been)\s+)?"
        r"(?:definitively\s+)?(?:confirmed|proven|established)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdiagnos\w*\b.{0,35}\b(?:with|as)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:confirm\w*|proven|prove\w*|diagnos\w*)\b.{{0,70}}"
        rf"\b{_DISEASE_REFERENCE}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_DISEASE_REFERENCE}\b.{{0,35}}"
        r"\b(?:is|was|has been|remains)\s+"
        r"(?:definitively\s+)?(?:confirmed|proven|diagnosed|present)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:this|the leaf|the plant|the image|the sample|it)\b.{{0,25}}"
        r"\b(?:definitely|certainly|clearly|visibly)?\s*"
        r"(?:has|shows|is infected with|is diseased with|is positive for)\b"
        rf".{{0,30}}\b{_DISEASE_REFERENCE}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:this|the leaf|the plant|the image|the sample|it)\b.{{0,15}}"
        r"\b(?:is|are|was|were)\s+"
        r"(?:definitely|certainly|clearly)?\s*(?:a|an|the)?\s*"
        rf"\b{_DISEASE_REFERENCE}\b",
        re.IGNORECASE,
    ),
)


def _reject_definitive_disease_confirmation(text: str, path: str) -> None:
    reviewable_text = text
    for pattern in _NEGATED_CONFIRMATION_PATTERNS:
        reviewable_text = pattern.sub(" ", reviewable_text)
    if any(pattern.search(reviewable_text) for pattern in _DEFINITIVE_DISEASE_PATTERNS):
        raise ValueError(
            f"{path} must not claim definitive disease confirmation from image "
            "or classifier evidence."
        )


def _validate_explanation_language(
    observations: list[dict[str, Any]],
    summary: str,
    claims: list[dict[str, Any]],
    path: str,
) -> None:
    for index, observation in enumerate(observations):
        _reject_definitive_disease_confirmation(
            observation["text"], f"{path}.observations[{index}].text"
        )
    _reject_definitive_disease_confirmation(summary, f"{path}.summary")
    for index, claim in enumerate(claims):
        _reject_definitive_disease_confirmation(
            claim["text"], f"{path}.claims[{index}].text"
        )


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> None:
    expected_type = schema["type"]
    type_matches = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: (
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
        ),
    }
    if expected_type not in type_matches or not type_matches[expected_type](value):
        raise ValueError(f"{path} must be {expected_type}.")

    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} must equal {schema['const']!r}.")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']!r}.")
    if "minLength" in schema and len(value) < schema["minLength"]:
        raise ValueError(f"{path} must not be empty.")
    if "minimum" in schema and value < schema["minimum"]:
        raise ValueError(f"{path} must be at least {schema['minimum']}.")
    if "maximum" in schema and value > schema["maximum"]:
        raise ValueError(f"{path} must be at most {schema['maximum']}.")
    if "minItems" in schema and len(value) < schema["minItems"]:
        raise ValueError(f"{path} must contain at least {schema['minItems']} items.")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        raise ValueError(f"{path} must contain at most {schema['maxItems']} items.")

    if expected_type == "object":
        properties = schema["properties"]
        missing = set(schema["required"]) - value.keys()
        if missing:
            raise ValueError(f"{path} is missing required fields: {sorted(missing)}.")
        extra = value.keys() - properties.keys()
        if schema.get("additionalProperties") is False and extra:
            raise ValueError(f"{path} has unsupported fields: {sorted(extra)}.")
        for key, item in value.items():
            if key in properties:
                _validate_value(item, properties[key], f"{path}.{key}")
    elif expected_type == "array":
        for index, item in enumerate(value):
            _validate_value(item, schema["items"], f"{path}[{index}]")


def validate_backend_response(response: Any) -> None:
    """Validate only the VLM-owned response using the shared output contract."""
    _validate_value(response, _BACKEND_RESPONSE_CONTRACT, "$.backend_response")
    _validate_explanation_language(
        response["observations"],
        response["summary"],
        response["claims"],
        "$.backend_response",
    )


def validate_output(
    output: Any,
    *,
    expected_class: str | None = None,
    expected_class_index: int | None = None,
    required_limitations: tuple[str, ...] = REQUIRED_LIMITATIONS,
) -> None:
    """Validate the strict output contract and classifier/Grad-CAM consistency."""
    _validate_value(output, OUTPUT_SCHEMA, "$")

    classifier = output["classifier"]
    visual_evidence = output["visual_evidence"]
    explanation = output["explanation"]
    _validate_explanation_language(
        visual_evidence["observations"],
        explanation["summary"],
        explanation["claims"],
        "$",
    )
    if visual_evidence["gradcam_target_class"] != classifier["predicted_class"]:
        raise ValueError("Grad-CAM target class must match the classifier prediction.")
    if (
        expected_class is not None
        and classifier["predicted_class"] != expected_class
    ):
        raise ValueError("Classifier prediction differs from the preserved prediction.")
    if (
        expected_class_index is not None
        and classifier["class_index"] != expected_class_index
    ):
        raise ValueError("Classifier index differs from the preserved class index.")
    missing_limitations = set(required_limitations) - set(output["limitations"])
    if missing_limitations:
        raise ValueError(
            f"Output is missing required limitations: {sorted(missing_limitations)}."
        )

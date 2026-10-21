"""Strict, dependency-free validation for Phase 9 structured explanations."""

from __future__ import annotations

import math
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

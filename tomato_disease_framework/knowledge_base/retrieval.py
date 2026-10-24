"""Deterministic local knowledge retrieval and recommendation policy."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlparse


KNOWLEDGE_BASE_VERSION = "1.0"
HEALTHY_CLASS_ID = "Tomato___healthy"
_CLASS_ID_PATTERN = re.compile(
    r"^Tomato___[A-Za-z0-9_-]+(?: [A-Za-z0-9_-]+)*$"
)
_VERIFICATION_METHOD = "manual_content_review"
_SUPPORT_STATUSES = {"sufficient", "insufficient", "irrelevant", "contradictory"}
_RESTRICTED_GUIDANCE = re.compile(
    r"\b(?:pesticide|fungicide|insecticide|herbicide|chemical|dosage|dose|"
    r"concentration|application rate|spray|spraying|apply|application)\b|"
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|ppm|%)\b",
    re.IGNORECASE,
)


class KnowledgeBaseError(ValueError):
    """Raised when the local knowledge base is malformed or unsafe to use."""


def _require_mapping(value: Any, path: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KnowledgeBaseError(f"{path} must be an object.")
    if set(value) != keys:
        raise KnowledgeBaseError(f"{path} must contain exactly {sorted(keys)}.")
    return value


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeBaseError(f"{path} must be a non-empty string.")
    return value.strip()


def _validate_source(source: Any, path: str) -> dict[str, Any]:
    keys = {
        "source_id",
        "title",
        "organization",
        "url",
        "publication_date",
        "verification_status",
        "verified_on",
        "verification_method",
    }
    item = dict(_require_mapping(source, path, keys))
    for field in ("source_id", "title", "organization", "url"):
        _require_nonempty_string(item[field], f"{path}.{field}")
    parsed_url = urlparse(item["url"])
    if (
        parsed_url.scheme != "https"
        or not parsed_url.netloc
        or parsed_url.username
        or parsed_url.password
    ):
        raise KnowledgeBaseError(f"{path}.url must be an HTTPS URL.")
    if not isinstance(item["publication_date"], str):
        raise KnowledgeBaseError(f"{path}.publication_date must be a string.")
    if item["publication_date"]:
        try:
            date.fromisoformat(item["publication_date"])
        except ValueError as error:
            raise KnowledgeBaseError(
                f"{path}.publication_date must be an ISO date or empty."
            ) from error
    if not isinstance(item["verified_on"], str) or not isinstance(
        item["verification_method"], str
    ):
        raise KnowledgeBaseError(
            f"{path} verification metadata must be strings."
        )
    status = item["verification_status"]
    if not isinstance(status, str) or status not in {"verified", "unverified"}:
        raise KnowledgeBaseError(
            f"{path}.verification_status must be verified or unverified."
        )
    if status == "verified":
        verified_on = _require_nonempty_string(
            item["verified_on"], f"{path}.verified_on"
        )
        try:
            date.fromisoformat(verified_on)
        except ValueError as error:
            raise KnowledgeBaseError(
                f"{path}.verified_on must be an ISO date."
            ) from error
        if item["verification_method"] != _VERIFICATION_METHOD:
            raise KnowledgeBaseError(
                f"{path}.verification_method must be {_VERIFICATION_METHOD!r}."
            )
    elif item["verified_on"] or item["verification_method"]:
        raise KnowledgeBaseError(
            f"{path} cannot contain verification details while unverified."
        )
    return item


def _validate_statements(
    values: Any,
    *,
    path: str,
    sources_by_id: Mapping[str, dict[str, Any]],
    guidance: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise KnowledgeBaseError(f"{path} must be an array.")
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    expected_keys = (
        {"guidance_id", "statement", "source_ids", "support_status"}
        if guidance
        else {"statement", "source_ids"}
    )
    for index, value in enumerate(values):
        item_path = f"{path}[{index}]"
        item = dict(_require_mapping(value, item_path, expected_keys))
        if guidance:
            identifier = _require_nonempty_string(
                item["guidance_id"], f"{item_path}.guidance_id"
            )
            if identifier in seen_ids:
                raise KnowledgeBaseError(f"{item_path}.guidance_id must be unique.")
            seen_ids.add(identifier)
            if (
                not isinstance(item["support_status"], str)
                or item["support_status"] not in _SUPPORT_STATUSES
            ):
                raise KnowledgeBaseError(
                    f"{item_path}.support_status is not a supported status."
                )
        _require_nonempty_string(item["statement"], f"{item_path}.statement")
        source_ids = item["source_ids"]
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or any(not isinstance(source_id, str) for source_id in source_ids)
            or len(source_ids) != len(set(source_ids))
        ):
            raise KnowledgeBaseError(
                f"{item_path}.source_ids must be a non-empty unique string array."
            )
        missing_sources = set(source_ids) - sources_by_id.keys()
        if missing_sources:
            raise KnowledgeBaseError(
                f"{item_path}.source_ids reference unknown sources: "
                f"{sorted(missing_sources)}."
            )
        validated.append(item)
    return validated


def validate_knowledge_base(data: Any) -> dict[str, Any]:
    """Validate the complete local knowledge-base document."""
    root = dict(_require_mapping(data, "$", {"schema_version", "records"}))
    if root["schema_version"] != KNOWLEDGE_BASE_VERSION:
        raise KnowledgeBaseError(
            f"$.schema_version must equal {KNOWLEDGE_BASE_VERSION!r}."
        )
    records = root["records"]
    if not isinstance(records, list) or not records:
        raise KnowledgeBaseError("$.records must be a non-empty array.")
    validated_records: list[dict[str, Any]] = []
    seen_classes: set[str] = set()
    seen_disease_ids: set[str] = set()
    record_keys = {
        "disease_id",
        "class_id",
        "display_name",
        "sources",
        "background",
        "reported_symptoms",
        "management_guidance",
    }
    for index, value in enumerate(records):
        path = f"$.records[{index}]"
        record = dict(_require_mapping(value, path, record_keys))
        disease_id = _require_nonempty_string(
            record["disease_id"], f"{path}.disease_id"
        )
        class_id = _require_nonempty_string(record["class_id"], f"{path}.class_id")
        if not _CLASS_ID_PATTERN.fullmatch(class_id):
            raise KnowledgeBaseError(f"{path}.class_id is not a tomato class ID.")
        _require_nonempty_string(record["display_name"], f"{path}.display_name")
        if class_id in seen_classes or disease_id in seen_disease_ids:
            raise KnowledgeBaseError(
                f"{path} duplicates a class_id or disease_id."
            )
        seen_classes.add(class_id)
        seen_disease_ids.add(disease_id)
        if not isinstance(record["sources"], list):
            raise KnowledgeBaseError(f"{path}.sources must be an array.")
        sources = [
            _validate_source(source, f"{path}.sources[{source_index}]")
            for source_index, source in enumerate(record["sources"])
        ]
        source_ids = [source["source_id"] for source in sources]
        if len(source_ids) != len(set(source_ids)):
            raise KnowledgeBaseError(f"{path}.sources contains duplicate source IDs.")
        sources_by_id = {source["source_id"]: source for source in sources}
        record["sources"] = sources
        record["background"] = _validate_statements(
            record["background"],
            path=f"{path}.background",
            sources_by_id=sources_by_id,
        )
        record["reported_symptoms"] = _validate_statements(
            record["reported_symptoms"],
            path=f"{path}.reported_symptoms",
            sources_by_id=sources_by_id,
        )
        record["management_guidance"] = _validate_statements(
            record["management_guidance"],
            path=f"{path}.management_guidance",
            sources_by_id=sources_by_id,
            guidance=True,
        )
        validated_records.append(record)
    root["records"] = validated_records
    return root


def load_knowledge_base(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the packaged knowledge base or an explicit JSON file."""
    source_path = (
        Path(path).expanduser()
        if path is not None
        else Path(__file__).with_name("tomato_diseases.json")
    )
    try:
        with source_path.open("r", encoding="utf-8-sig") as source_file:
            data = json.load(source_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KnowledgeBaseError(
            f"Could not load knowledge base from {source_path}."
        ) from error
    return validate_knowledge_base(data)


def retrieve_record(
    class_id: str,
    knowledge_base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an exact class match and evidence status; never substitute records."""
    if not isinstance(class_id, str) or not _CLASS_ID_PATTERN.fullmatch(class_id):
        raise ValueError("class_id must be a valid Tomato___ class identifier.")
    validated = (
        load_knowledge_base()
        if knowledge_base is None
        else validate_knowledge_base(knowledge_base)
    )
    for record in validated["records"]:
        if record["class_id"] == class_id:
            verified_count = sum(
                source["verification_status"] == "verified"
                for source in record["sources"]
            )
            sources_by_id = {
                source["source_id"]: source for source in record["sources"]
            }
            has_verified_relevant_guidance = any(
                guidance["support_status"] == "sufficient"
                and all(
                    sources_by_id[source_id]["verification_status"] == "verified"
                    for source_id in guidance["source_ids"]
                )
                and not _RESTRICTED_GUIDANCE.search(guidance["statement"])
                for guidance in record["management_guidance"]
            )
            return {
                "status": "record_found",
                "class_id": class_id,
                "evidence_status": (
                    "verified_relevant_sufficient_guidance"
                    if has_verified_relevant_guidance
                    else (
                        "verified_sources_not_sufficient_for_guidance"
                        if verified_count
                        else "no_verified_evidence"
                    )
                ),
                "record": deepcopy(record),
            }
    return {
        "status": "no_matching_record",
        "class_id": class_id,
        "evidence_status": "no_evidence",
        "record": None,
    }


def build_recommendations(
    predicted_class: str,
    knowledge_base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build application-owned recommendations only from verified local guidance."""
    result = retrieve_record(predicted_class, knowledge_base)
    base = {
        "basis_class": predicted_class,
        "items": [],
        "status": "withheld_no_verified_source",
        "withheld_reason": "no_matching_record",
    }
    if predicted_class == HEALTHY_CLASS_ID:
        base["withheld_reason"] = "healthy_class"
        return base
    if result["record"] is None:
        return base

    record = result["record"]
    base["withheld_reason"] = "no_sufficient_relevant_guidance"
    if any(
        guidance["support_status"] == "contradictory"
        for guidance in record["management_guidance"]
    ):
        base["withheld_reason"] = "contradictory_evidence"
        return base

    sources_by_id = {source["source_id"]: source for source in record["sources"]}
    recommendations: list[dict[str, Any]] = []
    restricted_guidance_found = False
    for guidance in record["management_guidance"]:
        if guidance["support_status"] != "sufficient":
            continue
        cited_sources = [sources_by_id[source_id] for source_id in guidance["source_ids"]]
        if any(source["verification_status"] != "verified" for source in cited_sources):
            continue
        if _RESTRICTED_GUIDANCE.search(guidance["statement"]):
            restricted_guidance_found = True
            continue
        recommendations.append(
            {
                "guidance_id": guidance["guidance_id"],
                "statement": guidance["statement"],
                "guidance_type": "general_educational_information",
                "sources": cited_sources,
            }
        )
    if not recommendations:
        if restricted_guidance_found:
            base["withheld_reason"] = "restricted_guidance"
        elif not any(
            source["verification_status"] == "verified"
            for source in record["sources"]
        ):
            base["withheld_reason"] = "no_verified_source"
        return base
    return {
        "basis_class": predicted_class,
        "items": recommendations,
        "status": "available_verified_source",
        "withheld_reason": "",
    }

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from knowledge_base.retrieval import (
    HEALTHY_CLASS_ID,
    KnowledgeBaseError,
    build_recommendations,
    load_knowledge_base,
    retrieve_record,
    validate_knowledge_base,
)
from vlm.schema import REQUIRED_LIMITATIONS, SCHEMA_VERSION, validate_output


CLASS_ID = "Tomato___Early_blight"
TEST_SOURCE = {
    "source_id": "test-source",
    "title": "Synthetic test-only source",
    "organization": "Unit test fixture",
    "url": "https://example.invalid/test-source",
    "publication_date": "",
    "verification_status": "verified",
    "verified_on": "2026-01-01",
    "verification_method": "manual_content_review",
}


def knowledge_base(
    *,
    source: dict[str, Any] | None = None,
    support_status: str = "sufficient",
    statement: str = "Use a test-only general educational statement.",
    class_id: str = CLASS_ID,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "records": [
            {
                "disease_id": "test_disease",
                "class_id": class_id,
                "display_name": "Test record",
                "sources": [] if source is None else [source],
                "background": [],
                "reported_symptoms": [],
                "management_guidance": (
                    []
                    if source is None
                    else [
                        {
                            "guidance_id": "test-guidance",
                            "statement": statement,
                            "source_ids": ["test-source"],
                            "support_status": support_status,
                        }
                    ]
                ),
            }
        ],
    }


def test_packaged_knowledge_base_has_ten_curated_records_with_valid_provenance():
    data = load_knowledge_base()

    expected_classes = {
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    }
    assert len(data["records"]) == len(expected_classes)
    assert {record["class_id"] for record in data["records"]} == expected_classes

    for record in data["records"]:
        sources_by_id = {source["source_id"]: source for source in record["sources"]}
        if record["class_id"] == HEALTHY_CLASS_ID:
            assert not record["sources"]
            assert not record["background"]
            assert not record["reported_symptoms"]
        else:
            assert sources_by_id
        for statement in (
            record["background"]
            + record["reported_symptoms"]
            + record["management_guidance"]
        ):
            assert statement["source_ids"]
            assert set(statement["source_ids"]) <= sources_by_id.keys()
        for source in record["sources"]:
            assert source["title"].strip()
            assert source["organization"].strip()
            assert source["url"].startswith("https://")
            assert source["verification_status"] in {"verified", "unverified"}
            assert source["verification_status"] == "unverified"
            assert source["verified_on"] == ""
            assert source["verification_method"] == ""

    assert all(not record["management_guidance"] for record in data["records"])
    result = retrieve_record(CLASS_ID)
    assert result["status"] == "record_found"
    assert result["evidence_status"] == "no_verified_evidence"
    assert result["record"]["class_id"] == CLASS_ID


def test_retrieval_matches_exact_class_and_reports_missing_class():
    data = knowledge_base(source=TEST_SOURCE)

    assert retrieve_record(CLASS_ID, data)["record"]["class_id"] == CLASS_ID
    missing = retrieve_record("Tomato___Target_Spot", data)
    assert missing == {
        "status": "no_matching_record",
        "class_id": "Tomato___Target_Spot",
        "evidence_status": "no_evidence",
        "record": None,
    }
    no_recommendation = build_recommendations("Tomato___Target_Spot", data)
    assert no_recommendation["items"] == []
    assert no_recommendation["withheld_reason"] == "no_matching_record"


def test_invalid_class_identifiers_and_malformed_knowledge_base_fail_safely():
    with pytest.raises(ValueError, match="valid Tomato"):
        retrieve_record("../Tomato___Early_blight", knowledge_base())

    invalid = knowledge_base(source=TEST_SOURCE)
    invalid["records"][0]["management_guidance"][0]["source_ids"] = [
        "missing-source"
    ]
    with pytest.raises(KnowledgeBaseError, match="unknown sources"):
        validate_knowledge_base(invalid)

    with pytest.raises(KnowledgeBaseError, match="exactly"):
        validate_knowledge_base({"schema_version": "1.0", "records": [], "extra": 1})


def test_recommendations_require_sufficient_relevant_guidance_and_verified_source():
    output = build_recommendations(CLASS_ID, knowledge_base(source=TEST_SOURCE))

    assert output["status"] == "available_verified_source"
    assert output["basis_class"] == CLASS_ID
    assert output["withheld_reason"] == ""
    assert output["items"] == [
        {
            "guidance_id": "test-guidance",
            "statement": "Use a test-only general educational statement.",
            "guidance_type": "general_educational_information",
            "sources": [TEST_SOURCE],
        }
    ]


@pytest.mark.parametrize(
    ("source", "support_status", "expected_reason"),
    [
        (None, "sufficient", "no_verified_source"),
        (
            {**TEST_SOURCE, "verification_status": "unverified",
             "verified_on": "", "verification_method": ""},
            "sufficient",
            "no_verified_source",
        ),
        (TEST_SOURCE, "irrelevant", "no_sufficient_relevant_guidance"),
        (TEST_SOURCE, "insufficient", "no_sufficient_relevant_guidance"),
    ],
)
def test_unverified_missing_or_irrelevant_evidence_withholds(
    source: dict[str, Any] | None,
    support_status: str,
    expected_reason: str,
):
    result = build_recommendations(
        CLASS_ID,
        knowledge_base(source=source, support_status=support_status),
    )

    assert result["status"] == "withheld_no_verified_source"
    assert result["items"] == []
    assert result["withheld_reason"] == expected_reason


def test_contradictory_evidence_withholds_even_with_sufficient_guidance():
    data = knowledge_base(source=TEST_SOURCE)
    data["records"][0]["management_guidance"].append(
        {
            "guidance_id": "contradiction",
            "statement": "Contradictory test fixture guidance.",
            "source_ids": ["test-source"],
            "support_status": "contradictory",
        }
    )

    result = build_recommendations(CLASS_ID, data)

    assert result["status"] == "withheld_no_verified_source"
    assert result["items"] == []
    assert result["withheld_reason"] == "contradictory_evidence"


def test_restricted_pesticide_or_dosage_guidance_is_withheld():
    result = build_recommendations(
        CLASS_ID,
        knowledge_base(
            source=TEST_SOURCE,
            statement="Apply pesticide at the recommended dosage.",
        ),
    )

    assert result["items"] == []
    assert result["withheld_reason"] == "restricted_guidance"


def test_healthy_class_never_receives_disease_treatment_recommendations():
    data = knowledge_base(source=TEST_SOURCE, class_id=HEALTHY_CLASS_ID)

    result = build_recommendations(HEALTHY_CLASS_ID, data)

    assert result["status"] == "withheld_no_verified_source"
    assert result["items"] == []
    assert result["withheld_reason"] == "healthy_class"


def test_policy_output_provenance_is_required_and_structurally_validated():
    output = {
        "schema_version": SCHEMA_VERSION,
        "classifier": {
            "model_id": "test-model",
            "predicted_class": CLASS_ID,
            "class_index": 0,
            "softmax_score": 0.75,
            "score_semantics": "uncalibrated_softmax_score",
        },
        "visual_evidence": {
            "gradcam_target_class": CLASS_ID,
            "gradcam_target_layer": "model.features[-1]",
            "observations": [
                {
                    "text": "A leaf is visible.",
                    "evidence_sources": ["input_image"],
                    "certainty": "direct_visual_observation",
                }
            ],
        },
        "explanation": {
            "summary": "The classifier prediction is not a diagnosis.",
            "claims": [
                {
                    "text": "The heatmap shows model attribution.",
                    "claim_type": "visual_observation",
                    "evidence_sources": ["gradcam_heatmap"],
                    "certainty": "supported",
                }
            ],
        },
        "recommendations": build_recommendations(
            CLASS_ID, knowledge_base(source=TEST_SOURCE)
        ),
        "limitations": list(REQUIRED_LIMITATIONS),
    }
    validate_output(output)

    output["recommendations"]["items"][0]["sources"][0][
        "verification_status"
    ] = "unverified"
    with pytest.raises(ValueError, match="must equal"):
        validate_output(output)


def test_policy_never_mutates_knowledge_base_fixture():
    source = deepcopy(TEST_SOURCE)
    data = knowledge_base(source=source)
    before = deepcopy(data)

    build_recommendations(CLASS_ID, data)

    assert data == before

"""Regression tests for the source-curated Phase 10 knowledge base.

Copy into the repository's tests/ directory after integrating the JSON file.
These tests validate schema/provenance wiring and fail-closed recommendation
behavior; they do not independently prove scientific validity of the sources.
"""

from knowledge_base import build_recommendations, load_knowledge_base, retrieve_record

EXPECTED_CLASSES = {
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


def test_curated_knowledge_base_covers_exact_expected_classes():
    kb = load_knowledge_base()
    assert {record["class_id"] for record in kb["records"]} == EXPECTED_CLASSES
    assert len(kb["records"]) == len(EXPECTED_CLASSES)


def test_each_statement_has_existing_source_provenance():
    kb = load_knowledge_base()
    for record in kb["records"]:
        source_ids = {source["source_id"] for source in record["sources"]}
        for statement in record["background"] + record["reported_symptoms"]:
            assert statement["source_ids"]
            assert set(statement["source_ids"]) <= source_ids
        for source in record["sources"]:
            assert source["verification_status"] in {"verified", "unverified"}
            assert source["verification_status"] == "unverified"
            assert source["verification_method"] == ""
            assert source["verified_on"] == ""


def test_no_management_recommendations_are_emitted_without_curated_guidance():
    kb = load_knowledge_base()
    for record in kb["records"]:
        assert record["management_guidance"] == []
        result = build_recommendations(record["class_id"], kb)
        assert result["items"] == []
        assert str(result["status"]).startswith("withheld_")


def test_unknown_class_is_not_mapped_to_a_known_record():
    result = retrieve_record("Tomato___Unknown_Disease")
    assert result["status"] == "no_matching_record"
    assert result["record"] is None

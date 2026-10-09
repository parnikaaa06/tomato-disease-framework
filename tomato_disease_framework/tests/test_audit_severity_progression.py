"""Tests for conservative severity/progression feasibility auditing."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.audit_severity_progression import build_audit, inspect_metadata_file


class SeverityProgressionAuditTests(unittest.TestCase):
    def test_disease_class_names_do_not_become_severity_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "external" / "color"
            dataset_root.mkdir(parents=True)
            (dataset_root / "Tomato___Early_blight").mkdir()
            (dataset_root / "Tomato___Early_blight" / "leaf.jpg").write_bytes(b"")

            report = build_audit(dataset_root, root)

        self.assertEqual(report["questions"]["disease_severity_ground_truth"], "NOT ESTABLISHED")
        self.assertEqual(report["questions"]["severity_grades_or_stages"], "NOT ESTABLISHED")
        self.assertEqual(report["conclusions"]["severity_or_progression_model_training"], "NOT JUSTIFIED")
        self.assertEqual(report["dataset_file_scan"]["image_count"], 1)

    def test_explicit_metadata_fields_are_candidate_evidence_not_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "plant_observations.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["plant_id", "observation_date", "severity_grade", "lesion_area_percent"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "plant_id": "plant-a",
                        "observation_date": "2025-01-01",
                        "severity_grade": "mild",
                        "lesion_area_percent": "12.5",
                    }
                )
                writer.writerow(
                    {
                        "plant_id": "plant-a",
                        "observation_date": "2025-01-08",
                        "severity_grade": "moderate",
                        "lesion_area_percent": "20.0",
                    }
                )

            item = inspect_metadata_file(path)

        self.assertEqual(item["status"], "inspected")
        self.assertIn("severity", item["candidate_field_groups"])
        self.assertIn("grade_or_stage", item["candidate_field_groups"])
        self.assertIn("disease_area", item["candidate_field_groups"])
        self.assertTrue(item["repeated_observation_check"]["repeated_observation_evidence"])
        self.assertEqual(item["repeated_observation_check"]["identity_groups_with_multiple_dates"], 1)


if __name__ == "__main__":
    unittest.main()

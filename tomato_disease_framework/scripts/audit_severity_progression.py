"""Audit whether available metadata supports severity or progression research.

This is a read-only feasibility audit. It never derives severity from disease
class names, estimates affected area from pixels, or treats filename similarity
as repeated observation of the same plant or leaf.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
STRUCTURED_EXTENSIONS = {".csv", ".tsv", ".json", ".yaml", ".yml"}
DISCOVERY_TERMS = {
    "severity",
    "grade",
    "stage",
    "progression",
    "temporal",
    "longitudinal",
    "metadata",
    "annotation",
    "label",
    "manifest",
    "observation",
    "plant",
    "leaf",
    "capture",
    "visit",
    "split",
    "groundtruth",
}
ANNOTATION_DIRECTORY_TERMS = {
    "severity",
    "grade",
    "stage",
    "progression",
    "temporal",
    "longitudinal",
    "metadata",
    "annotation",
    "label",
    "observation",
    "capture",
    "visit",
    "groundtruth",
}
FIELD_GROUPS: dict[str, set[str]] = {
    "severity": {
        "severity",
        "severityscore",
        "severityvalue",
        "diseaseseverity",
        "severitygrade",
        "severitylevel",
        "severitypercent",
        "severitypercentage",
    },
    "grade_or_stage": {
        "grade",
        "severitygrade",
        "diseasegrade",
        "stage",
        "diseasestage",
        "severitystage",
        "severitylevel",
    },
    "disease_area": {
        "lesionarea",
        "diseasearea",
        "affectedarea",
        "infectedarea",
        "lesionfraction",
        "diseaseareapercent",
        "lesionareapercent",
        "affectedareapercent",
        "affectedfraction",
    },
    "plant_or_leaf_identity": {
        "plantid",
        "plantidentifier",
        "leafid",
        "leafidentifier",
        "subjectid",
        "specimenid",
        "individualplantid",
        "individualleafid",
    },
    "observation_time": {
        "timestamp",
        "datetime",
        "capturedat",
        "capturedate",
        "acquisitiondate",
        "observationdate",
        "observedat",
        "visitdate",
        "collectiondate",
        "imagingdate",
    },
}
IDENTITY_FIELDS = FIELD_GROUPS["plant_or_leaf_identity"]
TIME_FIELDS = FIELD_GROUPS["observation_time"]
CURATED_REPO_FILES = (
    Path("data/dataset_report/annotation_availability.json"),
    Path("data/dataset_report/dataset_report.json"),
    Path("data/dataset_report/image_manifest.csv"),
    Path("data/splits/train.csv"),
    Path("data/splits/validation.csv"),
    Path("data/splits/test.csv"),
)


def normalize_field_name(name: str) -> str:
    """Normalize metadata keys for exact, conservative field-name matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def matching_groups(field_names: Iterable[str]) -> dict[str, list[str]]:
    """Return recognized candidate metadata fields, without inferring their meaning."""
    matched: dict[str, list[str]] = {}
    for group, accepted_names in FIELD_GROUPS.items():
        fields = sorted(
            field
            for field in field_names
            if normalize_field_name(field.split(".")[-1]) in accepted_names
        )
        if fields:
            matched[group] = fields
    return matched


def _flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested JSON/YAML objects and list records into dotted keys."""
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_mapping(item, path))
    elif isinstance(value, list):
        mappings = [item for item in value if isinstance(item, dict)]
        if mappings:
            for item in mappings:
                flattened.update(_flatten_mapping(item, prefix))
        elif prefix:
            flattened[prefix] = value
    elif prefix:
        flattened[prefix] = value
    return flattened


def _read_tabular(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def _read_structured(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return _read_tabular(path)

    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle) if suffix == ".json" else yaml.safe_load(handle)

    if isinstance(data, list):
        records = [row for row in data if isinstance(row, dict)]
        flattened = [_flatten_mapping(row) for row in records]
    elif isinstance(data, dict):
        records_value = next(
            (
                value
                for value in data.values()
                if isinstance(value, list) and value and all(isinstance(row, dict) for row in value)
            ),
            None,
        )
        if records_value is not None:
            flattened = [_flatten_mapping(row) for row in records_value]
        else:
            flattened = [_flatten_mapping(data)]
    else:
        flattened = []

    fields = sorted({key for row in flattened for key in row})
    return fields, flattened


def _is_populated(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "nan", "None", "null"}


def _parse_time(value: Any) -> str | None:
    """Normalize common date/time values to date strings for repeat checks."""
    if not _is_populated(value):
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    for pattern in ("%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            continue
    return None


def _repeat_observation_evidence(
    rows: list[dict[str, Any]], field_names: list[str]
) -> dict[str, Any]:
    """Count explicit IDs with multiple distinct dates; do not use filenames as IDs."""
    normalized_to_field = {normalize_field_name(field): field for field in field_names}
    id_fields = [normalized_to_field[name] for name in IDENTITY_FIELDS if name in normalized_to_field]
    time_fields = [normalized_to_field[name] for name in TIME_FIELDS if name in normalized_to_field]
    if not id_fields or not time_fields:
        return {
            "identity_fields": id_fields,
            "time_fields": time_fields,
            "records_with_identity_and_time": 0,
            "identity_groups_with_multiple_dates": 0,
            "repeated_observation_evidence": False,
        }

    observations: defaultdict[str, set[str]] = defaultdict(set)
    records_with_pair = 0
    for row in rows:
        identity = next((row.get(field) for field in id_fields if _is_populated(row.get(field))), None)
        time_value = next((row.get(field) for field in time_fields if _is_populated(row.get(field))), None)
        normalized_date = _parse_time(time_value)
        if _is_populated(identity) and normalized_date is not None:
            records_with_pair += 1
            observations[str(identity).strip()].add(normalized_date)

    repeated_groups = sum(1 for dates in observations.values() if len(dates) > 1)
    return {
        "identity_fields": id_fields,
        "time_fields": time_fields,
        "records_with_identity_and_time": records_with_pair,
        "identity_groups_with_multiple_dates": repeated_groups,
        "repeated_observation_evidence": repeated_groups > 0,
    }


def _is_discovery_candidate(path: Path) -> bool:
    text = path.name.lower()
    return any(term in text for term in DISCOVERY_TERMS)


def discover_metadata_files(dataset_root: Path | None, repo_root: Path) -> list[Path]:
    """Find relevant metadata documents plus authoritative project manifests."""
    discovered: set[Path] = set()
    roots = [repo_root / path for path in CURATED_REPO_FILES]
    discovered.update(path.resolve() for path in roots if path.is_file())

    if dataset_root is not None and dataset_root.is_dir():
        # Search the image root only to a shallow depth for sidecar metadata.
        for current, dirs, files in os.walk(dataset_root):
            current_path = Path(current)
            depth = len(current_path.relative_to(dataset_root).parts)
            if depth >= 2:
                dirs[:] = []
            for filename in files:
                path = current_path / filename
                if path.suffix.lower() in STRUCTURED_EXTENSIONS and _is_discovery_candidate(path):
                    discovered.add(path.resolve())

        # Inspect sibling annotation/metadata directories, not the entire Drive.
        sibling_root = dataset_root.parent
        for child in sorted(sibling_root.iterdir()):
            if not child.is_dir() or not any(
                term in child.name.lower() for term in ANNOTATION_DIRECTORY_TERMS
            ):
                continue
            for current, dirs, files in os.walk(child):
                current_path = Path(current)
                depth = len(current_path.relative_to(child).parts)
                if depth >= 3:
                    dirs[:] = []
                for filename in files:
                    path = current_path / filename
                    if path.suffix.lower() in STRUCTURED_EXTENSIONS and _is_discovery_candidate(path):
                        discovered.add(path.resolve())

        for path in sibling_root.iterdir():
            if path.is_file() and path.suffix.lower() in STRUCTURED_EXTENSIONS and _is_discovery_candidate(path):
                discovered.add(path.resolve())

    return sorted(discovered)


def inspect_metadata_file(path: Path) -> dict[str, Any]:
    """Summarize schema and populated candidate fields without exporting values."""
    try:
        fields, rows = _read_structured(path)
        groups = matching_groups(fields)
        populated_by_group = {
            group: sum(
                1
                for row in rows
                if any(_is_populated(row.get(field)) for field in group_fields)
            )
            for group, group_fields in groups.items()
        }
        repeated = _repeat_observation_evidence(rows, fields)
        return {
            "source_path": str(path),
            "file_type": path.suffix.lower(),
            "status": "inspected",
            "record_count": len(rows),
            "fields": fields,
            "candidate_field_groups": groups,
            "records_with_populated_candidate_fields": populated_by_group,
            "repeated_observation_check": repeated,
            "error": None,
        }
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError, yaml.YAMLError, ValueError) as exc:
        return {
            "source_path": str(path),
            "file_type": path.suffix.lower(),
            "status": "unreadable",
            "record_count": None,
            "fields": [],
            "candidate_field_groups": {},
            "records_with_populated_candidate_fields": {},
            "repeated_observation_check": _repeat_observation_evidence([], []),
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_audit(dataset_root: Path | None, repo_root: Path) -> dict[str, Any]:
    dataset_available = dataset_root is not None and dataset_root.is_dir()
    metadata_files = discover_metadata_files(dataset_root if dataset_available else None, repo_root)
    inventory = [inspect_metadata_file(path) for path in metadata_files]

    observed_fields: defaultdict[str, list[str]] = defaultdict(list)
    populated_groups: set[str] = set()
    repeated_evidence: list[str] = []
    for item in inventory:
        for group, fields in item["candidate_field_groups"].items():
            if item["records_with_populated_candidate_fields"].get(group, 0) > 0:
                populated_groups.add(group)
                observed_fields[group].extend(fields)
        repeat_check = item["repeated_observation_check"]
        if repeat_check["repeated_observation_evidence"]:
            repeated_evidence.append(item["source_path"])

    source_images = 0
    class_counts: Counter[str] = Counter()
    if dataset_available:
        for path in dataset_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                source_images += 1
                try:
                    class_counts[path.relative_to(dataset_root).parts[0]] += 1
                except (IndexError, ValueError):
                    class_counts["(root)"] += 1

    inspected_roots = ["repository metadata/split artifacts"]
    if dataset_available:
        inspected_roots.insert(0, "configured image root and selected sibling metadata directories")
    else:
        inspected_roots.append("external dataset metadata not inspected because TDF_DATASET_ROOT is missing or invalid")

    # Candidate metadata columns are evidence for review, not proof of annotation
    # validity; this project has not recorded provenance or a validated label protocol.
    questions = {
        "disease_severity_ground_truth": "NOT ESTABLISHED",
        "severity_grades_or_stages": "NOT ESTABLISHED",
        "disease_area_measurements": "NOT ESTABLISHED",
        "temporal_progression": "NOT ESTABLISHED",
        "repeated_same_plant_or_leaf_observations": "NOT ESTABLISHED",
    }
    candidate_field_evidence = {
        group: {
            "fields": sorted(set(fields)),
            "records_with_populated_values_found": group in populated_groups,
        }
        for group, fields in observed_fields.items()
    }
    conclusions = {
        "severity_target": "NOT ESTABLISHED",
        "severity_grades_or_stages": "NOT ESTABLISHED",
        "disease_area_measurements": "NOT ESTABLISHED",
        "temporal_progression": "NOT ESTABLISHED",
        "repeated_plant_or_leaf_observations": "NOT ESTABLISHED",
        "severity_or_progression_model_training": "NOT JUSTIFIED",
    }
    if "severity" in populated_groups or "grade_or_stage" in populated_groups:
        conclusions["severity_target"] = "CANDIDATE FIELDS FOUND — TARGET VALIDITY NOT ESTABLISHED"
    if "grade_or_stage" in populated_groups:
        conclusions["severity_grades_or_stages"] = "CANDIDATE FIELDS FOUND — LABEL VALIDITY NOT ESTABLISHED"
    if "disease_area" in populated_groups:
        conclusions["disease_area_measurements"] = "CANDIDATE FIELDS FOUND — MEASUREMENT VALIDITY NOT ESTABLISHED"
    if "observation_time" in populated_groups:
        conclusions["temporal_progression"] = "TIME FIELDS FOUND — LINKED PROGRESSION NOT ESTABLISHED"
    if repeated_evidence:
        conclusions["repeated_plant_or_leaf_observations"] = (
            "REPEATED ID/DATE RECORDS FOUND — PLANT/LEAF IDENTITY REQUIRES VALIDATION"
        )

    return {
        "dataset_root": str(dataset_root) if dataset_root is not None else "NOT SET",
        "dataset_root_status": "available" if dataset_available else "not available / not verified",
        "dataset_file_scan": {
            "status": "completed" if dataset_available else "not performed",
            "image_count": source_images if dataset_available else None,
            "class_counts": dict(sorted(class_counts.items())) if dataset_available else {},
        },
        "inspected_scope": inspected_roots,
        "metadata_inventory_count": len(inventory),
        "metadata_inventory": inventory,
        "candidate_field_evidence": candidate_field_evidence,
        "questions": questions,
        "conclusions": conclusions,
        "evidence_and_limitations": [
            "Plant disease class labels identify categories, not disease severity, severity grade, or disease stage.",
            "Image appearance and lesion-like pixels were not converted into severity scores or affected-area measurements.",
            "Filename similarity and duplicate-image records were not treated as repeated observations of the same plant or leaf.",
            "A field name or populated metadata value is only a discovery lead; annotation provenance, definitions, and protocol must be validated before establishing ground truth.",
            "The audit does not train a severity model or infer temporal progression.",
        ],
        "manual_review_required": True,
        "repeated_observation_candidate_sources": repeated_evidence,
        "summary": (
            "Defensible ground truth for severity, grade/stage, disease-area measurement, temporal "
            "progression, and repeated observations of the same plant or leaf has not been established "
            "from inspected sources. Severity/progression model training is not justified."
        ),
    }


def write_inventory(path: Path, inventory: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_path",
        "file_type",
        "status",
        "record_count",
        "fields",
        "candidate_field_groups",
        "records_with_populated_candidate_fields",
        "repeated_observation_check",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in inventory:
            writer.writerow(
                {
                    **item,
                    "fields": "; ".join(item["fields"]),
                    "candidate_field_groups": json.dumps(item["candidate_field_groups"], sort_keys=True),
                    "records_with_populated_candidate_fields": json.dumps(
                        item["records_with_populated_candidate_fields"], sort_keys=True
                    ),
                    "repeated_observation_check": json.dumps(
                        item["repeated_observation_check"], sort_keys=True
                    ),
                }
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ["TDF_DATASET_ROOT"]).expanduser()
        if os.environ.get("TDF_DATASET_ROOT")
        else None,
        help="External PlantVillage image root; defaults to TDF_DATASET_ROOT.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root used to inspect tracked dataset metadata and split files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase7"),
        help="Directory for the audit report and candidate metadata inventory.",
    )
    args = parser.parse_args(argv)

    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root else None
    if dataset_root is not None and not dataset_root.is_dir():
        print(
            f"Dataset root is unavailable or not a directory: {dataset_root}. "
            "The audit will still inspect repository metadata.",
            file=sys.stderr,
        )

    repo_root = args.repo_root.resolve()
    report = build_audit(dataset_root, repo_root)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    inventory_path = output_dir / "severity_progression_candidate_inventory.csv"
    report_path = output_dir / "severity_progression_feasibility_report.yaml"

    output_dir.mkdir(parents=True, exist_ok=True)
    write_inventory(inventory_path, report["metadata_inventory"])
    with report_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=True)

    print(f"Dataset root status: {report['dataset_root_status']}")
    print(f"Structured metadata sources inspected: {report['metadata_inventory_count']}")
    print(f"Candidate inventory written to: {inventory_path}")
    print(f"Feasibility report written to: {report_path}")
    print(report["summary"])
    for name, status in report["conclusions"].items():
        print(f"  {name}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
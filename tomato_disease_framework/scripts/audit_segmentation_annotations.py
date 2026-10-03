"""Read-only audit of possible segmentation/mask annotations.

This script is intentionally conservative: it does not assume that any directory
named "segmented" or "mask" is a verified disease-lesion annotation dataset. It
only audits candidate files and reports whether they are technically compatible
with segmentation-style masks and whether source-to-mask pairing exists.

The script is designed to run from the repository root in Colab or a local
Python environment where TDF_DATASET_ROOT points at the PlantVillage dataset.
It never modifies, renames, moves, converts, or generates dataset images.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MANUAL_VISUAL_REVIEW = {
    "evidence_source": (
        "User-provided visual inspection of representative original/segmented pairs "
        "and the Phase 6 final review."
    ),
    "candidate_collection": "PlantVillage segmented collection",
    "candidate_collection_present": True,
    "corresponding_original_pairs_available": True,
    "leaf_background_segmentation_confirmed": True,
    "observations": [
        "The transformation removes or reduces background while retaining the complete leaf.",
        "Disease symptoms remain within the retained leaf region.",
        "The observed images represent leaf/background segmentation, not verified disease-lesion masks.",
        "No explicit disease-lesion annotation source was established.",
    ],
    "disease_lesion_ground_truth": "NOT ESTABLISHED",
}
CANDIDATE_KEYWORDS = (
    "segmented",
    "segmentation",
    "mask",
    "masks",
    "lesion",
    "lesions",
    "annotation",
    "annotations",
    "leaf",
    "background",
)
REMOVE_STEM_TOKENS = (
    "segmented",
    "segmentation",
    "mask",
    "masks",
    "binary",
    "annotation",
    "annotations",
    "lesion",
    "lesions",
    "leaf",
    "background",
)


def normalize_stem(name: str) -> str:
    """Normalize a filename stem to support conservative pairing heuristics."""
    normalized = Path(name).stem.lower().replace("-", "_").replace(" ", "_")
    for token in REMOVE_STEM_TOKENS:
        normalized = normalized.replace(f"_{token}", "_").replace(f"{token}_", "_")
        normalized = normalized.replace(token, "")
    normalized = "".join(ch for ch in normalized if ch.isalnum() or ch in "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def candidate_name_match(path: Path) -> bool:
    """Return True if the path name suggests a candidate segmentation resource."""
    text = " ".join(path.parts).lower()
    return any(term in text for term in CANDIDATE_KEYWORDS)


def iter_image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_candidate_collections(dataset_root: Path) -> list[Path]:
    """Search likely parent directories around the dataset and collect candidate dirs."""
    search_roots: list[Path] = []
    for candidate in [dataset_root, dataset_root.parent, dataset_root.parent.parent]:
        if candidate not in search_roots and candidate.exists():
            search_roots.append(candidate)

    collected: set[Path] = set()
    for search_root in search_roots:
        for path in sorted(search_root.rglob("*")):
            if path.is_dir() and any(term in path.name.lower() for term in CANDIDATE_KEYWORDS):
                collected.add(path)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                if any(term in path.name.lower() for term in CANDIDATE_KEYWORDS):
                    collected.add(path.parent)
    return sorted(collected)


def get_class_names(dataset_root: Path) -> set[str]:
    class_names: set[str] = set()
    if not dataset_root.exists():
        return class_names
    for path in dataset_root.iterdir():
        if path.is_dir():
            class_names.add(path.name)
    return class_names


def sample_image_metrics(image_path: Path) -> dict[str, Any]:
    """Compute lightweight pixel metadata without modifying the source data."""
    try:
        with Image.open(image_path) as image:
            image.load()
            mode = image.mode
            width, height = image.size
            bands = len(image.getbands())
            array = np.asarray(image)
    except Exception as exc:  # pragma: no cover - dataset-file dependent
        return {
            "status": "unreadable",
            "error": str(exc),
            "mode": None,
            "width": None,
            "height": None,
            "bands": None,
            "binary_like": False,
            "unique_pixel_values": None,
            "foreground_fraction": None,
            "background_fraction": None,
        }

    flattened = array.reshape(-1)
    unique_values = np.unique(flattened)
    unique_count = int(unique_values.size)
    binary_like = False
    foreground_fraction = None
    background_fraction = None

    if array.ndim == 2 or (array.ndim == 3 and array.shape[-1] == 1):
        flat = np.asarray(array).reshape(-1)
        unique_count = int(np.unique(flat).size)
        if unique_count <= 32:
            binary_like = True
        if flat.size:
            threshold = float(np.max(flat) * 0.5) if np.max(flat) > 0 else 0.0
            foreground_fraction = float(np.mean(flat > threshold))
            background_fraction = 1.0 - foreground_fraction
    elif array.ndim == 3:
        if unique_count <= 32:
            binary_like = True
        if array.shape[-1] in {3, 4} and array.size:
            gray = np.mean(array, axis=-1)
            threshold = float(np.max(gray) * 0.5) if np.max(gray) > 0 else 0.0
            foreground_fraction = float(np.mean(gray > threshold))
            background_fraction = 1.0 - foreground_fraction

    return {
        "status": "ok",
        "error": None,
        "mode": mode,
        "width": width,
        "height": height,
        "bands": bands,
        "binary_like": binary_like,
        "unique_pixel_values": unique_count,
        "foreground_fraction": foreground_fraction,
        "background_fraction": background_fraction,
    }


def analyze_candidate_collection(collection: Path, source_root: Path, class_names: set[str]) -> dict[str, Any]:
    files = sorted(
        path
        for path in collection.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        return {
            "path": str(collection),
            "file_count": 0,
            "file_extensions": [],
            "widths": [],
            "heights": [],
            "modes": [],
            "bands": [],
            "binary_like_count": 0,
            "sample_unique_values": [],
            "sample_foreground_background": [],
            "class_counts": {},
            "notes": "No qualifying image files were found in this candidate collection.",
        }

    modes: list[str] = []
    widths: list[int] = []
    heights: list[int] = []
    bands: list[int] = []
    binary_like_count = 0
    unique_values_samples: list[int] = []
    foreground_background_samples: list[dict[str, float | str]] = []
    class_counts: Counter[str] = Counter()

    sample_files = files[: min(len(files), 8)]
    for file_path in sample_files:
        metrics = sample_image_metrics(file_path)
        if metrics.get("status") == "ok":
            modes.append(metrics["mode"])
            widths.append(metrics["width"])
            heights.append(metrics["height"])
            bands.append(metrics["bands"])
            unique_values_samples.append(metrics["unique_pixel_values"])
            if metrics.get("binary_like"):
                binary_like_count += 1
            if metrics.get("foreground_fraction") is not None:
                foreground_background_samples.append(
                    {
                        "file": file_path.name,
                        "foreground_fraction": float(metrics["foreground_fraction"]),
                        "background_fraction": float(metrics["background_fraction"]),
                    }
                )

    for file_path in files:
        rel_parts = list(file_path.relative_to(source_root).parts) if file_path.is_relative_to(source_root) else list(file_path.parts)
        for part in rel_parts:
            if part in class_names:
                class_counts[part] += 1
                break
        if not class_counts and file_path.parts:
            for part in file_path.parts:
                if part in class_names:
                    class_counts[part] += 1
                    break

    return {
        "path": str(collection),
        "file_count": len(files),
        "file_extensions": sorted({path.suffix.lower() for path in files}),
        "widths": sorted(set(widths)),
        "heights": sorted(set(heights)),
        "modes": sorted(set(modes)),
        "bands": sorted(set(bands)),
        "binary_like_count": binary_like_count,
        "sample_unique_values": unique_values_samples,
        "sample_foreground_background": foreground_background_samples,
        "class_counts": dict(sorted(class_counts.items())),
        "notes": "Binary-like appearance does not by itself confirm disease-lesion masks; file names and folder labels are not enough.",
    }


def pair_source_and_candidates(dataset_root: Path, candidate_files: list[Path]) -> dict[str, Any]:
    source_files = iter_image_files(dataset_root)
    source_by_stem: defaultdict[str, list[Path]] = defaultdict(list)
    for source_path in source_files:
        if source_path.name:
            source_by_stem[normalize_stem(source_path.name)].append(source_path)

    candidate_by_stem: defaultdict[str, list[Path]] = defaultdict(list)
    for candidate in candidate_files:
        candidate_by_stem[normalize_stem(candidate.name)].append(candidate)

    matched_pairs: list[tuple[Path, Path]] = []
    candidate_unmatched: list[Path] = []
    source_unmatched: set[Path] = set(source_files)
    duplicate_stems: set[str] = set()
    filename_mismatches: list[str] = []

    for candidate in candidate_files:
        stem_key = normalize_stem(candidate.name)
        if len(source_by_stem.get(stem_key, [])) > 1:
            duplicate_stems.add(stem_key)
        matches = source_by_stem.get(stem_key, [])
        if matches:
            matched_pairs.append((candidate, matches[0]))
            source_unmatched.discard(matches[0])
        else:
            candidate_unmatched.append(candidate)
            filename_mismatches.append(f"{candidate.name} -> no direct source stem match")

    for stem_key, source_paths in source_by_stem.items():
        if len(source_paths) > 1:
            duplicate_stems.add(stem_key)
        if stem_key not in candidate_by_stem and source_paths:
            filename_mismatches.append(f"source stem {stem_key} has no candidate counterpart")

    return {
        "total_color_images": len(source_files),
        "total_candidate_annotation_files": len(candidate_files),
        "matched_pairs": len(matched_pairs),
        "unmatched_candidate_files": len(candidate_unmatched),
        "unmatched_source_images": len(source_unmatched),
        "duplicate_stems": sorted(duplicate_stems),
        "obvious_filename_mismatches": filename_mismatches[:25],
        "matched_pair_examples": [
            {"candidate": str(candidate), "source": str(source)}
            for candidate, source in matched_pairs[:10]
        ],
    }


def write_inventory_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "path",
        "file_count",
        "file_extensions",
        "widths",
        "heights",
        "modes",
        "bands",
        "binary_like_count",
        "sample_unique_values",
        "sample_foreground_background",
        "class_counts",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_report_yaml(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False, default_flow_style=False)


def decide_statuses(details: dict[str, Any]) -> dict[str, str]:
    """Return evidence-based statuses for the required audit questions."""
    candidate_count = details.get("candidate_count", 0)
    matched_pairs = details.get("matched_pairs", 0)
    binary_like = details.get("binary_like_count", 0)
    class_coverage = details.get("class_coverage_count", 0)
    has_real_lesion_evidence = bool(details.get("lesion_evidence_verified", False))
    manual_review = details.get("manual_visual_review", {})
    dataset_inspected = details.get("dataset_root_status") == "exists"

    candidate_present = candidate_count > 0 or manual_review.get("candidate_collection_present", False)
    pairs_available = matched_pairs > 0 or manual_review.get(
        "corresponding_original_pairs_available", False
    )
    leaf_background_confirmed = bool(manual_review.get("leaf_background_segmentation_confirmed"))

    status_a = "YES" if candidate_present else ("NO" if dataset_inspected else "NOT ESTABLISHED")
    status_b = (
        "YES (representative pairs confirmed)"
        if pairs_available
        else ("NO" if dataset_inspected else "NOT ESTABLISHED")
    )
    status_c = (
        "YES (leaf/background only)"
        if leaf_background_confirmed
        else ("YES — binary-like candidate images" if binary_like > 0 else "NOT ESTABLISHED")
    )
    status_d = "YES" if has_real_lesion_evidence else "NOT ESTABLISHED"
    status_e = "YES" if class_coverage > 0 and has_real_lesion_evidence else "NOT ESTABLISHED"
    status_f = (
        "YES"
        if has_real_lesion_evidence and pairs_available and class_coverage > 0
        else "NOT ESTABLISHED"
    )
    status_g = (
        "JUSTIFIED"
        if has_real_lesion_evidence and pairs_available and class_coverage > 0 and candidate_present
        else "NOT JUSTIFIED"
    )

    return {
        "A": status_a,
        "B": status_b,
        "C": status_c,
        "D": status_d,
        "E": status_e,
        "F": status_f,
        "G": status_g,
    }


def build_report(dataset_root: Path | None, source_root: Path | None) -> dict[str, Any]:
    if dataset_root is None or not dataset_root.exists():
        questions = decide_statuses(
            {
                "dataset_root_status": "missing",
                "manual_visual_review": MANUAL_VISUAL_REVIEW,
            }
        )
        return {
            "dataset_root": str(dataset_root) if dataset_root is not None else "NOT SET",
            "dataset_root_status": "missing",
            "audit_summary": {
                "status": "read-only audit could not inspect dataset files because TDF_DATASET_ROOT is missing or invalid.",
                "source_color_images": 0,
                "candidate_annotation_files": 0,
                "matched_pairs": 0,
            },
            "questions": questions,
            "evidence": [
                "TDF_DATASET_ROOT was not set or the configured directory does not exist.",
                "No file-level inspection was performed because no valid source dataset root was available.",
                "Manual visual review findings below were supplied for this PlantVillage dataset; they are distinct from this run's unavailable file-level inspection.",
            ],
            "candidate_collections": [],
            "manual_visual_review": MANUAL_VISUAL_REVIEW,
            "conclusion": (
                "Original-to-segmented pairs are available, and visual inspection indicates "
                "leaf/background segmentation that retains the complete leaf. Disease-lesion "
                "annotations were not established. U-Net disease-lesion training is not justified."
            ),
            "decision": {
                "original_segmented_pairing": "AVAILABLE",
                "leaf_background_segmentation": "AVAILABLE",
                "verified_disease_lesion_masks": "NOT ESTABLISHED",
                "disease_lesion_segmentation_dataset": "NOT AVAILABLE",
                "unet_disease_lesion_training": "NOT JUSTIFIED",
            },
            "notes": [
                "This report intentionally does not infer disease-lesion labels from directory names or binary-like image appearance.",
            ],
        }

    source_root = dataset_root.resolve()
    source_images = iter_image_files(source_root)
    class_names = get_class_names(source_root)
    candidate_collections = discover_candidate_collections(source_root)
    candidate_files: list[Path] = []
    collection_rows: list[dict[str, Any]] = []

    for collection in candidate_collections:
        collection_data = analyze_candidate_collection(collection, source_root, class_names)
        collection_rows.append(collection_data)
        candidate_files.extend(
            path
            for path in collection.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    pairing = pair_source_and_candidates(source_root, candidate_files)
    binary_like_total = sum(int(row.get("binary_like_count", 0)) for row in collection_rows)
    class_coverage_count = sum(
        int(count)
        for row in collection_rows
        for count in row.get("class_counts", {}).values()
    )

    all_questions = decide_statuses(
        {
            "dataset_root_status": "exists",
            "candidate_count": len(candidate_files),
            "matched_pairs": pairing["matched_pairs"],
            "binary_like_count": binary_like_total,
            "class_coverage_count": class_coverage_count,
            "lesion_evidence_verified": False,
            "manual_visual_review": MANUAL_VISUAL_REVIEW,
        }
    )

    conclusion = (
        "Original-to-segmented pairs are available. Visual inspection of representative pairs "
        "shows background removal or reduction while retaining the complete leaf; disease "
        "symptoms remain within the retained leaf. The available images represent "
        "leaf/background segmentation, not verified disease-lesion masks. No explicit "
        "disease-lesion annotation source was established. U-Net disease-lesion training is "
        "not justified using the currently established annotations."
    )

    report = {
        "dataset_root": str(source_root),
        "dataset_root_status": "exists",
        "audit_summary": {
            "status": "read-only audit complete",
            "source_color_images": len(source_images),
            "candidate_annotation_files": len(candidate_files),
            "matched_pairs": pairing["matched_pairs"],
            "unmatched_candidate_files": pairing["unmatched_candidate_files"],
            "unmatched_source_images": pairing["unmatched_source_images"],
            "duplicate_stems": pairing["duplicate_stems"],
            "obvious_filename_mismatches": pairing["obvious_filename_mismatches"],
            "candidate_collections_detected": len(collection_rows),
        },
        "questions": all_questions,
        "manual_visual_review": MANUAL_VISUAL_REVIEW,
        "evidence": [
            "Candidate collections were discovered by name and path inspection only; names were never treated as proof of disease-lesion masks.",
            "The script checked file counts, extensions, dimensions, image mode, sampled pixel statistics, and candidate-to-source pairing compatibility.",
            "Binary-like images were flagged as binary-like only when the sampled pixel distribution was sparse; binary appearance alone does not confirm disease-region labels.",
            "Manual visual review found leaf/background segmentation in representative original/segmented pairs; this is not disease-lesion ground truth.",
            "No explicit disease-lesion ground truth source was established.",
        ],
        "candidate_collections": collection_rows,
        "pairing": pairing,
        "conclusion": conclusion,
        "decision": {
            "original_segmented_pairing": "AVAILABLE",
            "leaf_background_segmentation": "AVAILABLE",
            "verified_disease_lesion_masks": "NOT ESTABLISHED",
            "disease_lesion_segmentation_dataset": "NOT AVAILABLE",
            "unet_disease_lesion_training": "NOT JUSTIFIED",
        },
        "notes": [
            "This audit is intentionally conservative and does not infer disease lesions from segmentation-like foreground/background regions.",
            "Manual visual inspection is required to confirm whether any candidate images correspond to lesion annotations versus leaf/background segmentation.",
        ],
    }
    return report


def main() -> int:
    dataset_root_value = os.environ.get("TDF_DATASET_ROOT")
    dataset_root = Path(dataset_root_value).expanduser().resolve() if dataset_root_value else None

    if dataset_root is None:
        print("TDF_DATASET_ROOT is not set.")
        print("The audit will produce a conservative NOT ESTABLISHED report without inspecting dataset files.")
    elif not dataset_root.exists():
        print(f"TDF_DATASET_ROOT does not exist: {dataset_root}")
        print("The audit will produce a conservative NOT ESTABLISHED report without inspecting dataset files.")
    elif not dataset_root.is_dir():
        print(f"TDF_DATASET_ROOT is not a directory: {dataset_root}")
        print("The audit will produce a conservative NOT ESTABLISHED report without inspecting dataset files.")
    else:
        print(f"Inspecting dataset root: {dataset_root}")

    report = build_report(dataset_root, dataset_root)

    results_dir = Path("results/phase6")
    results_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = results_dir / "segmentation_candidate_inventory.csv"
    report_yaml_path = results_dir / "segmentation_feasibility_report.yaml"

    inventory_rows: list[dict[str, Any]] = []
    for collection in report.get("candidate_collections", []):
        inventory_rows.append(
            {
                "path": collection.get("path", ""),
                "file_count": collection.get("file_count", 0),
                "file_extensions": "; ".join(collection.get("file_extensions", [])),
                "widths": "; ".join(str(value) for value in collection.get("widths", [])),
                "heights": "; ".join(str(value) for value in collection.get("heights", [])),
                "modes": "; ".join(collection.get("modes", [])),
                "bands": "; ".join(str(value) for value in collection.get("bands", [])),
                "binary_like_count": collection.get("binary_like_count", 0),
                "sample_unique_values": "; ".join(str(v) for v in collection.get("sample_unique_values", [])),
                "sample_foreground_background": str(collection.get("sample_foreground_background", [])),
                "class_counts": str(collection.get("class_counts", {})),
                "notes": collection.get("notes", ""),
            }
        )
    write_inventory_csv(inventory_path, inventory_rows)
    write_report_yaml(report_yaml_path, report)

    print(f"Inventory written to: {inventory_path}")
    print(f"Report written to: {report_yaml_path}")
    print("\nAudit conclusion:")
    print(report.get("conclusion", "No conclusion available."))
    print("\nEvidence-based answers:")
    for key in ["A", "B", "C", "D", "E", "F", "G"]:
        print(f"  {key}: {report['questions'][key]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

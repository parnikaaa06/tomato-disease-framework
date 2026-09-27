"""Phase 1 dataset inspection driven by the project's config.yaml.

This script never creates annotations, severity labels, or derived masks. It
reports what can be verified from the configured dataset root and emits
explicit unavailable reports when that root is not mounted in the current
environment (for example, when config.yaml points to a Colab Drive path).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit("PyYAML is required to run Phase 1 inspection.") from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit("Pillow is required to run Phase 1 inspection.") from exc


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
MASK_DIRECTORY_NAMES = {"mask", "masks", "segmentation", "segmentations", "annotations"}
SEVERITY_TERMS = {"severity", "severity_label", "severity_labels", "grade", "rating"}


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def unavailable_reports(
    report_dir: Path, configured_path: str, reason: str
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    status = "not available / not verified"
    summary = {
        "status": status,
        "configured_dataset_path": configured_path,
        "resolved_dataset_path": str(Path(configured_path)),
        "reason": reason,
        "verified_at_runtime": False,
        "note": "No dataset statistics, annotations, masks, severity labels, or model results were fabricated.",
    }
    (report_dir / "dataset_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (report_dir / "annotation_availability.json").write_text(
        json.dumps(
            {
                "status": status,
                "segmentation_masks": {
                    "available": False,
                    "verified": False,
                    "reason": reason,
                },
                "severity_annotations": {
                    "available": False,
                    "verified": False,
                    "reason": reason,
                },
                "classification_labels_are_not_masks_or_severity": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(
        report_dir / "dataset_summary.csv",
        ["metric", "value", "status"],
        [{"metric": key, "value": value, "status": status} for key, value in summary.items()],
    )
    write_csv(
        report_dir / "class_distribution.csv",
        ["class_name", "image_count", "percentage", "status"],
        [{"class_name": "", "image_count": "", "percentage": "", "status": status}],
    )
    write_csv(
        report_dir / "image_statistics.csv",
        ["metric", "value", "status"],
        [{"metric": "all image statistics", "value": "", "status": status}],
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dataset(dataset_root: Path, report_dir: Path, configured_path: str) -> None:
    files = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file()
    ]
    image_files = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    class_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    class_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    dimensions: list[tuple[int, int]] = []
    unreadable: list[dict[str, str]] = []
    hashes: defaultdict[str, list[str]] = defaultdict(list)

    for path in image_files:
        relative_parts = path.relative_to(dataset_root).parts
        class_name = relative_parts[0] if len(relative_parts) > 1 else "(root)"
        class_counts[class_name] += 1
        extension_counts[path.suffix.lower()] += 1
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions.append(image.size)
            hashes[sha256(path)].append(str(path.relative_to(dataset_root)))
        except (OSError, ValueError) as exc:
            unreadable.append({"path": str(path.relative_to(dataset_root)), "error": str(exc)})

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    total = len(image_files)
    percentages = {
        name: (count / total * 100 if total else 0.0)
        for name, count in class_counts.items()
    }
    min_count = min(class_counts.values()) if class_counts else 0
    max_count = max(class_counts.values()) if class_counts else 0
    imbalance_ratio = max_count / min_count if min_count else None

    mask_candidates = [
        str(path.relative_to(dataset_root))
        for path in files
        if any(part.lower() in MASK_DIRECTORY_NAMES for part in path.relative_to(dataset_root).parts)
    ]
    severity_candidates = [
        str(path.relative_to(dataset_root))
        for path in files
        if any(term in path.name.lower() for term in SEVERITY_TERMS)
        or any(term in part.lower() for part in path.relative_to(dataset_root).parts for term in SEVERITY_TERMS)
    ]

    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        report_dir / "class_distribution.csv",
        ["class_name", "image_count", "percentage"],
        [
            {"class_name": name, "image_count": count, "percentage": round(percentages[name], 6)}
            for name, count in sorted(class_counts.items())
        ],
    )
    write_csv(
        report_dir / "image_statistics.csv",
        ["metric", "value"],
        [
            {"metric": "image_count", "value": total},
            {"metric": "readable_image_count", "value": len(dimensions)},
            {"metric": "corrupted_or_unreadable_count", "value": len(unreadable)},
            {"metric": "min_width", "value": min((width for width, _ in dimensions), default=None)},
            {"metric": "max_width", "value": max((width for width, _ in dimensions), default=None)},
            {"metric": "min_height", "value": min((height for _, height in dimensions), default=None)},
            {"metric": "max_height", "value": max((height for _, height in dimensions), default=None)},
            {"metric": "unique_resolution_count", "value": len(Counter(dimensions))},
            {"metric": "exact_duplicate_groups", "value": len(duplicate_groups)},
        ],
    )
    write_csv(
        report_dir / "dataset_summary.csv",
        ["metric", "value"],
        [
            {"metric": "configured_dataset_path", "value": configured_path},
            {"metric": "class_directory_count", "value": len(class_dirs)},
            {"metric": "image_count", "value": total},
            {"metric": "extension_counts", "value": json.dumps(dict(extension_counts), sort_keys=True)},
            {"metric": "imbalance_ratio_max_to_min", "value": imbalance_ratio},
        ],
    )
    (report_dir / "dataset_report.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "configured_dataset_path": configured_path,
                "class_directories": [path.name for path in class_dirs],
                "tomato_classes": [
                    path.name for path in class_dirs if path.name.lower().startswith("tomato")
                ],
                "class_counts": dict(sorted(class_counts.items())),
                "class_percentages": percentages,
                "imbalance": {
                    "max_class_count": max_count,
                    "min_class_count": min_count,
                    "max_to_min_ratio": imbalance_ratio,
                },
                "extensions": dict(sorted(extension_counts.items())),
                "dimensions": {
                    "readable_image_count": len(dimensions),
                    "min_width": min((width for width, _ in dimensions), default=None),
                    "max_width": max((width for width, _ in dimensions), default=None),
                    "min_height": min((height for _, height in dimensions), default=None),
                    "max_height": max((height for _, height in dimensions), default=None),
                    "resolution_frequencies": {
                        f"{width}x{height}": count
                        for (width, height), count in Counter(dimensions).items()
                    },
                },
                "corrupted_or_unreadable_files": unreadable,
                "exact_duplicate_groups": duplicate_groups,
                "near_duplicates": {
                    "status": "not assessed",
                    "reason": "No perceptual-hash dependency is enabled in the Phase 1 environment.",
                },
                "annotation_candidates": {
                    "mask_path_candidates": mask_candidates,
                    "severity_path_or_filename_candidates": severity_candidates,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (report_dir / "annotation_availability.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "segmentation_masks": {
                    "available": bool(mask_candidates),
                    "verified": False,
                    "candidate_count": len(mask_candidates),
                    "reason": "Candidates require manual/semantic validation; class labels are not masks.",
                },
                "severity_annotations": {
                    "available": bool(severity_candidates),
                    "verified": False,
                    "candidate_count": len(severity_candidates),
                    "reason": "Candidates require semantic validation; disease class labels are not severity labels.",
                },
                "classification_labels_are_not_masks_or_severity": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    configured_path = str(config["data"]["external_dataset_path"]).strip()
    report_dir = Path(config["data"]["dataset_report_dir"])
    dataset_root = Path(os.path.expandvars(os.path.expanduser(configured_path)))

    if not configured_path:
        unavailable_reports(report_dir, configured_path, "external_dataset_path is empty.")
        return
    if not dataset_root.exists() or not dataset_root.is_dir():
        unavailable_reports(
            report_dir,
            configured_path,
            "Configured dataset root is not mounted or is not a directory in this runtime.",
        )
        return
    inspect_dataset(dataset_root, report_dir, configured_path)


if __name__ == "__main__":
    main()

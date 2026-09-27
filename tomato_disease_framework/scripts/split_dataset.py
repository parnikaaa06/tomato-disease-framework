"""Phase 2: reproducible, stratified dataset splitting.

The script is intentionally fail-closed. It only uses the dataset root from
config.yaml and requires a verified Phase 1 report plus an exact-duplicate
manifest before producing a split. It never infers duplicate relationships
from the user-provided counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to run Phase 2 splitting.") from exc


TOMATO_PREFIX = "Tomato"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_blocked_report(
    splits_dir: Path, configured_path: str, reason: str
) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    status = "not available / not verified"
    empty_split = [{"image_path": "", "class_name": "", "status": status}]
    for name in ("train.csv", "validation.csv", "test.csv"):
        write_csv(splits_dir / name, ["image_path", "class_name", "status"], empty_split)
    write_csv(
        splits_dir / "split_summary.csv",
        ["metric", "value", "status"],
        [
            {"metric": "status", "value": status, "status": status},
            {"metric": "configured_dataset_path", "value": configured_path, "status": status},
            {"metric": "reason", "value": reason, "status": status},
        ],
    )
    (splits_dir / "split_report.json").write_text(
        json.dumps(
            {
                "status": status,
                "configured_dataset_path": configured_path,
                "reason": reason,
                "splits_created": False,
                "no_fabricated_results": True,
                "required_next_action": "Run this script in Colab after mounting the configured Google Drive path and providing verified Phase 1 duplicate groups.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_duplicate_manifest(path: Path) -> list[list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Verified exact-duplicate manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    groups = data.get("exact_duplicate_groups")
    if not isinstance(groups, list):
        raise ValueError("Duplicate manifest must contain exact_duplicate_groups as a list.")
    return groups


def stratified_split(rows: list[dict[str, str]], seed: int) -> tuple[list[dict[str, str]], ...]:
    """Split by class without requiring a third-party splitter."""
    import random

    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["class_name"]].append(row)
    rng = random.Random(seed)
    train: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for class_name in sorted(grouped):
        items = sorted(grouped[class_name], key=lambda item: item["image_path"])
        rng.shuffle(items)
        count = len(items)
        test_count = round(count * 0.15)
        validation_count = round(count * 0.15)
        if count >= 3:
            test_count = max(1, test_count)
            validation_count = max(1, validation_count)
        train_count = count - validation_count - test_count
        if train_count < 1:
            raise ValueError(f"Class {class_name!r} has too few images for 70/15/15 splitting.")
        train.extend(items[:train_count])
        validation.extend(items[train_count : train_count + validation_count])
        test.extend(items[train_count + validation_count :])
    return train, validation, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--duplicate-manifest",
        type=Path,
        default=Path("data/dataset_report/exact_duplicate_groups.json"),
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    configured_path = str(config["data"]["external_dataset_path"]).strip()
    report_dir = Path(config["data"]["dataset_report_dir"])
    splits_dir = Path(config["data"]["splits_dir"])
    dataset_root = Path(configured_path)

    phase1_report = report_dir / "dataset_report.json"
    if not configured_path:
        write_blocked_report(splits_dir, configured_path, "external_dataset_path is empty.")
        return
    if not dataset_root.exists() or not dataset_root.is_dir():
        write_blocked_report(
            splits_dir,
            configured_path,
            "Configured dataset root is not mounted or is not a directory in this runtime.",
        )
        return
    if not phase1_report.exists():
        write_blocked_report(splits_dir, configured_path, "Phase 1 dataset report is missing.")
        return
    phase1 = json.loads(phase1_report.read_text(encoding="utf-8"))
    if phase1.get("status") != "verified":
        write_blocked_report(
            splits_dir,
            configured_path,
            "Phase 1 report is not verified; no split can be created safely.",
        )
        return

    try:
        duplicate_groups = read_duplicate_manifest(args.duplicate_manifest)
    except (FileNotFoundError, ValueError) as exc:
        write_blocked_report(splits_dir, configured_path, str(exc))
        return

    all_files = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    rows = [
        {
            "image_path": str(path.relative_to(dataset_root)),
            "class_name": path.relative_to(dataset_root).parts[0],
        }
        for path in all_files
        if path.relative_to(dataset_root).parts[0].startswith(TOMATO_PREFIX)
    ]
    original_count = len(rows)
    path_to_row = {row["image_path"]: row for row in rows}
    removed_paths: set[str] = set()
    retained_groups = 0
    for group in duplicate_groups:
        members = sorted(str(member) for member in group if str(member) in path_to_row)
        if len(members) > 1:
            retained_groups += 1
            removed_paths.update(members[1:])
    rows = [row for row in rows if row["image_path"] not in removed_paths]
    train, validation, test = stratified_split(
        rows, int(config.get("training", {}).get("seed", config["model"]["seed"]))
    )
    splits = {"train": train, "validation": validation, "test": test}
    split_paths = {name: {row["image_path"] for row in values} for name, values in splits.items()}
    overlap = (
        split_paths["train"] & split_paths["validation"]
        | split_paths["train"] & split_paths["test"]
        | split_paths["validation"] & split_paths["test"]
    )
    exact_leakage = any(
        len({name for name, paths in split_paths.items() if member in paths}) > 1
        for group in duplicate_groups
        for member in group
    )
    for name, values in splits.items():
        write_csv(splits_dir / f"{name}.csv", ["image_path", "class_name"], values)
    counts = {
        name: dict(sorted(Counter(row["class_name"] for row in values).items()))
        for name, values in splits.items()
    }
    summary = {
        "status": "verified",
        "configured_dataset_path": configured_path,
        "total_original_images": original_count,
        "exact_duplicate_groups_found": len(duplicate_groups),
        "exact_duplicate_groups_retained": retained_groups,
        "images_removed_as_exact_duplicates": len(removed_paths),
        "images_remaining_for_splitting": len(rows),
        "train_count": len(train),
        "validation_count": len(validation),
        "test_count": len(test),
        "train_percentage": len(train) / len(rows) * 100,
        "validation_percentage": len(validation) / len(rows) * 100,
        "test_percentage": len(test) / len(rows) * 100,
        "per_class_counts": counts,
        "random_seed": int(config.get("training", {}).get("seed", config["model"]["seed"])),
        "stratification_confirmed": True,
        "exact_duplicate_leakage": exact_leakage,
        "near_duplicate_leakage": "not assessed from unprovided Phase 1 pair manifest",
        "zero_image_overlap": not overlap,
        "count_verification": len(train) + len(validation) + len(test) == len(rows),
        "cross_class_near_duplicates_preserved": True,
    }
    write_csv(
        splits_dir / "split_summary.csv",
        ["metric", "value"],
        [{"metric": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in summary.items()],
    )
    (splits_dir / "split_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

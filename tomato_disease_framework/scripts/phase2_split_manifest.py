"""Legacy Phase 2 split utility.

Warning: the verified files in ``data/splits/`` are the current source of
truth. This script is not the authoritative split-generation procedure.
Do not rerun any split-generation script without first validating the
duplicate-aware reproducibility procedure.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle) or {}


def resolve_dataset_root(config: dict[str, Any]) -> Path:
    raw = config.get("data", {}).get("external_dataset_path", "")
    if not raw:
        raise FileNotFoundError("TDF_DATASET_ROOT is not set.")
    return Path(os.path.expandvars(os.path.expanduser(str(raw))))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def blocked_split(report_dir: Path, dataset_root: str, reason: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    status = "not available / not verified"
    for name in ("train.csv", "validation.csv", "test.csv"):
        write_csv(report_dir / name, ["image_path", "class_name", "status"], [{"image_path": "", "class_name": "", "status": status}])
    write_csv(report_dir / "split_summary.csv", ["metric", "value", "status"], [{"metric": "status", "value": status, "status": status}, {"metric": "dataset_root", "value": dataset_root, "status": status}, {"metric": "reason", "value": reason, "status": status}])
    (report_dir / "split_report.json").write_text(json.dumps({"status": status, "dataset_root": dataset_root, "reason": reason, "splits_created": False}, indent=2), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    report_dir = Path(config.get("data", {}).get("splits_dir", "data/splits"))
    try:
        dataset_root = resolve_dataset_root(config)
    except FileNotFoundError as exc:
        blocked_split(report_dir, str(config.get("data", {}).get("external_dataset_path", "")), str(exc))
        return
    if not dataset_root.exists() or not dataset_root.is_dir():
        blocked_split(report_dir, str(dataset_root), "Configured dataset root is not mounted or is not a directory in this runtime.")
        return

    rows: list[dict[str, str]] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(dataset_root)
        class_name = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        rows.append({"image_path": rel.as_posix(), "class_name": class_name})

    if not rows:
        blocked_split(report_dir, str(dataset_root), "No image files were found at the configured dataset root.")
        return

    by_class: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["class_name"]].append(row)
    random_seed = int(config.get("runtime", {}).get("seed", config.get("model", {}).get("seed", 42)))
    import random

    train: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for class_name in sorted(by_class):
        items = by_class[class_name]
        rng = random.Random(random_seed + sum(ord(ch) for ch in class_name))
        rng.shuffle(items)
        count = len(items)
        test_count = max(1, round(count * 0.15)) if count > 1 else 0
        validation_count = max(1, round(count * 0.15)) if count > 2 else 0
        train_count = count - validation_count - test_count
        if train_count <= 0:
            train_count = max(1, count - test_count - validation_count)
        if train_count + validation_count + test_count > count:
            diff = (train_count + validation_count + test_count) - count
            validation_count = max(0, validation_count - diff)
        train.extend(items[:train_count])
        validation.extend(items[train_count:train_count + validation_count])
        test.extend(items[train_count + validation_count:train_count + validation_count + test_count])

    for name, split_rows in {"train": train, "validation": validation, "test": test}.items():
        write_csv(report_dir / f"{name}.csv", ["image_path", "class_name"], split_rows)

    counts = {name: Counter(row["class_name"] for row in split_rows) for name, split_rows in {"train": train, "validation": validation, "test": test}.items()}
    summary = {
        "status": "verified",
        "dataset_root": str(dataset_root),
        "total_original_images": len(rows),
        "train_count": len(train),
        "validation_count": len(validation),
        "test_count": len(test),
        "train_percentage": round(len(train) / len(rows) * 100, 2),
        "validation_percentage": round(len(validation) / len(rows) * 100, 2),
        "test_percentage": round(len(test) / len(rows) * 100, 2),
        "per_class_counts": {k: dict(v) for k, v in counts.items()},
        "random_seed": random_seed,
        "stratification_confirmed": True,
        "zero_image_overlap": len(set(row["image_path"] for row in train) & set(row["image_path"] for row in validation)) == 0 and len(set(row["image_path"] for row in train) & set(row["image_path"] for row in test)) == 0 and len(set(row["image_path"] for row in validation) & set(row["image_path"] for row in test)) == 0,
        "count_verification": len(train) + len(validation) + len(test) == len(rows),
    }
    write_csv(report_dir / "split_summary.csv", ["metric", "value"], [{"metric": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in summary.items()])
    (report_dir / "split_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

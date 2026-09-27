from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle) or {}


def resolve_dataset_root(config: dict[str, Any]) -> Path:
    raw = config.get("data", {}).get("external_dataset_path", "")
    if not raw:
        raise FileNotFoundError("TDF_DATASET_ROOT is not configured in config.yaml.")
    resolved = os.path.expandvars(os.path.expanduser(str(raw)))
    return Path(resolved)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    image = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(image.getdata())
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            current = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits.append(1 if current > right else 0)
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return format(value, f"0{hash_size * hash_size}b")


def hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def blocked_manifest(report_dir: Path, dataset_root: str, reason: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    status = "not available / not verified"
    write_csv(
        report_dir / "image_manifest.csv",
        ["relative_file_path", "class_name", "file_name", "extension", "width", "height", "file_size_bytes", "md5_hash", "exact_duplicate_group_id", "near_duplicate_status", "status"],
        [{"relative_file_path": "", "class_name": "", "file_name": "", "extension": "", "width": "", "height": "", "file_size_bytes": "", "md5_hash": "", "exact_duplicate_group_id": "", "near_duplicate_status": "", "status": status}],
    )
    write_csv(
        report_dir / "duplicate_groups.csv",
        ["group_id", "duplicate_type", "number_of_files", "class_information", "member_file_paths", "status"],
        [{"group_id": "", "duplicate_type": "", "number_of_files": "", "class_information": "", "member_file_paths": "", "status": status}],
    )
    write_csv(
        report_dir / "near_duplicate_review.csv",
        ["pair_id", "file_a", "file_b", "distance", "same_class", "review_status", "status"],
        [{"pair_id": "", "file_a": "", "file_b": "", "distance": "", "same_class": "", "review_status": "", "status": status}],
    )
    (report_dir / "manifest_status.json").write_text(
        json.dumps({
            "status": status,
            "configured_dataset_path": dataset_root,
            "resolved_dataset_path": str(Path(dataset_root)),
            "reason": reason,
            "note": "No raw-image manifest was fabricated because the dataset is not available in this runtime.",
        }, indent=2),
        encoding="utf-8",
    )


def build_manifest(config_path: str | Path) -> None:
    config = load_config(config_path)
    dataset_root = resolve_dataset_root(config)
    report_dir = Path(config.get("data", {}).get("dataset_report_dir", "data/dataset_report"))
    if not dataset_root.exists() or not dataset_root.is_dir():
        blocked_manifest(report_dir, str(dataset_root), "Configured dataset root is not mounted or is not a directory in this runtime.")
        return

    image_rows: list[dict[str, Any]] = []
    md5_to_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(dataset_root)
        class_name = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            width, height = "", ""
        digest = sha256(path)
        row = {
            "relative_file_path": str(rel).replace('\\', '/'),
            "class_name": class_name,
            "file_name": path.name,
            "extension": path.suffix.lower(),
            "width": width,
            "height": height,
            "file_size_bytes": path.stat().st_size,
            "md5_hash": digest,
            "exact_duplicate_group_id": "EXD_NONE",
            "near_duplicate_status": "not evaluated",
            "status": "verified",
        }
        md5_to_rows[digest].append(row)
        image_rows.append(row)

    group_map: dict[str, str] = {}
    group_entries: list[dict[str, Any]] = []
    for idx, (_, rows) in enumerate(sorted(md5_to_rows.items(), key=lambda item: item[0]), start=1):
        group_id = f"EXD_{idx:05d}"
        for row in rows:
            row["exact_duplicate_group_id"] = group_id
        if len(rows) > 1:
            group_entries.append({
                "group_id": group_id,
                "duplicate_type": "exact_duplicate",
                "number_of_files": len(rows),
                "class_information": ", ".join(sorted({r["class_name"] for r in rows})),
                "member_file_paths": "; ".join(r["relative_file_path"] for r in rows),
                "status": "verified",
            })

    # Perceptual hash near-duplicate review without automatic deletion.
    perceptual_map: dict[str, str] = {}
    for row in image_rows:
        try:
            with Image.open(dataset_root / row["relative_file_path"].replace('/', os.sep)) as image:
                perceptual_map[row["relative_file_path"]] = dhash(image)
        except Exception:
            perceptual_map[row["relative_file_path"]] = ""

    near_rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, left_path in enumerate(sorted(perceptual_map)):
        left_hash = perceptual_map[left_path]
        if not left_hash:
            continue
        left_row = next(r for r in image_rows if r["relative_file_path"] == left_path)
        for right_path in sorted(perceptual_map)[index + 1:]:
            right_hash = perceptual_map[right_path]
            if not right_hash:
                continue
            pair = tuple(sorted((left_path, right_path)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            distance = hamming_distance(left_hash, right_hash)
            if distance <= 4:
                right_row = next(r for r in image_rows if r["relative_file_path"] == right_path)
                same_class = left_row["class_name"] == right_row["class_name"]
                near_rows.append({
                    "pair_id": f"ND_{len(near_rows)+1:05d}",
                    "file_a": left_path,
                    "file_b": right_path,
                    "distance": distance,
                    "same_class": str(same_class),
                    "review_status": "candidate_near_duplicate",
                    "status": "verified",
                })

    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        report_dir / "image_manifest.csv",
        ["relative_file_path", "class_name", "file_name", "extension", "width", "height", "file_size_bytes", "md5_hash", "exact_duplicate_group_id", "near_duplicate_status", "status"],
        image_rows,
    )
    write_csv(
        report_dir / "duplicate_groups.csv",
        ["group_id", "duplicate_type", "number_of_files", "class_information", "member_file_paths", "status"],
        group_entries,
    )
    write_csv(
        report_dir / "near_duplicate_review.csv",
        ["pair_id", "file_a", "file_b", "distance", "same_class", "review_status", "status"],
        near_rows,
    )

    manifest_report = {
        "status": "verified",
        "dataset_root": str(dataset_root),
        "total_images": len(image_rows),
        "unique_content_hashes": len(md5_to_rows),
        "exact_duplicate_groups": len(group_entries),
        "images_in_exact_duplicate_groups": sum(item["number_of_files"] for item in group_entries),
        "near_duplicate_candidate_pairs": len(near_rows),
        "policy": {
            "exact_duplicates_grouped_by_md5": True,
            "raw_images_left_intact": True,
            "near_duplicates_not_auto_deleted": True,
            "cross_class_near_duplicates_preserved": True,
        },
    }
    (report_dir / "manifest_status.json").write_text(json.dumps(manifest_report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    try:
        build_manifest(args.config)
    except FileNotFoundError as exc:
        config = load_config(args.config)
        report_dir = Path(config.get("data", {}).get("dataset_report_dir", "data/dataset_report"))
        blocked_manifest(report_dir, str(config.get("data", {}).get("external_dataset_path", "")), str(exc))
    

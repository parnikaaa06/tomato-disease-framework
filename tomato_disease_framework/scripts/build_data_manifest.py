"""Build the Phase 1 image manifest and duplicate reports.

This is a report-building utility, not the authoritative split-generation
procedure. The verified files in ``data/splits/`` are the current source of
truth. Do not rerun any split-generation script without first validating the
duplicate-aware reproducibility procedure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

NEAR_DUPLICATE_THRESHOLD = 4


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle) or {}


def resolve_dataset_root(config: dict[str, Any]) -> Path:
    raw = config.get("data", {}).get("external_dataset_path", "")

    if not raw:
        raise FileNotFoundError(
            "TDF_DATASET_ROOT is not configured in config.yaml."
        )

    resolved = os.path.expandvars(os.path.expanduser(str(raw)))
    return Path(resolved)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    image = image.convert("L").resize(
        (hash_size + 1, hash_size),
        Image.Resampling.LANCZOS,
    )

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


def hamming_distance_int(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def blocked_manifest(
    report_dir: Path,
    dataset_root: str,
    reason: str,
) -> None:

    report_dir.mkdir(parents=True, exist_ok=True)

    status = "not available / not verified"

    write_csv(
        report_dir / "image_manifest.csv",
        [
            "relative_file_path",
            "class_name",
            "file_name",
            "extension",
            "width",
            "height",
            "file_size_bytes",
            "sha256_hash",
            "exact_duplicate_group_id",
            "near_duplicate_status",
            "status",
        ],
        [
            {
                "relative_file_path": "",
                "class_name": "",
                "file_name": "",
                "extension": "",
                "width": "",
                "height": "",
                "file_size_bytes": "",
                "sha256_hash": "",
                "exact_duplicate_group_id": "",
                "near_duplicate_status": "",
                "status": status,
            }
        ],
    )

    write_csv(
        report_dir / "duplicate_groups.csv",
        [
            "group_id",
            "duplicate_type",
            "number_of_files",
            "class_information",
            "member_file_paths",
            "status",
        ],
        [
            {
                "group_id": "",
                "duplicate_type": "",
                "number_of_files": "",
                "class_information": "",
                "member_file_paths": "",
                "status": status,
            }
        ],
    )

    write_csv(
        report_dir / "near_duplicate_review.csv",
        [
            "pair_id",
            "file_a",
            "file_b",
            "distance",
            "same_class",
            "review_status",
            "status",
        ],
        [
            {
                "pair_id": "",
                "file_a": "",
                "file_b": "",
                "distance": "",
                "same_class": "",
                "review_status": "",
                "status": status,
            }
        ],
    )

    (report_dir / "manifest_status.json").write_text(
        json.dumps(
            {
                "status": status,
                "configured_dataset_path": dataset_root,
                "resolved_dataset_path": str(Path(dataset_root)),
                "reason": reason,
                "note": (
                    "No raw-image manifest was fabricated because "
                    "the dataset is not available in this runtime."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


class BKTree:
    """
    BK-tree for efficient nearest-neighbor search
    using Hamming distance on binary perceptual hashes.
    """

    def __init__(self) -> None:
        self.root: tuple[int, str] | None = None
        self.children: dict[int, dict[int, tuple[int, str]]] = defaultdict(dict)

    def add(self, value: int, path: str) -> None:
        if self.root is None:
            self.root = (value, path)
            return

        current_value, current_path = self.root

        while True:
            distance = hamming_distance_int(value, current_value)

            if distance == 0:
                return

            if distance not in self.children[current_value]:
                self.children[current_value][distance] = (value, path)
                return

            current_value, current_path = self.children[current_value][distance]

    def search(
        self,
        value: int,
        max_distance: int,
    ) -> list[tuple[int, str, int]]:

        if self.root is None:
            return []

        results: list[tuple[int, str, int]] = []

        stack = [self.root]

        while stack:
            current_value, current_path = stack.pop()

            distance = hamming_distance_int(value, current_value)

            if distance <= max_distance:
                results.append(
                    (current_value, current_path, distance)
                )

            lower = distance - max_distance
            upper = distance + max_distance

            for edge_distance, child in self.children[current_value].items():
                if lower <= edge_distance <= upper:
                    stack.append(child)

        return results


def build_manifest(config_path: str | Path) -> None:

    config = load_config(config_path)

    dataset_root = resolve_dataset_root(config)

    report_dir = Path(
        config.get("data", {}).get(
            "dataset_report_dir",
            "data/dataset_report",
        )
    )

    if not dataset_root.exists() or not dataset_root.is_dir():

        blocked_manifest(
            report_dir,
            str(dataset_root),
            "Configured dataset root is not mounted or is not a directory in this runtime.",
        )

        return

    print(f"Dataset root: {dataset_root}")
    print("Scanning image files...")

    image_paths = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Images discovered: {len(image_paths)}")

    image_rows: list[dict[str, Any]] = []

    sha256_to_rows: defaultdict[
        str,
        list[dict[str, Any]]
    ] = defaultdict(list)

    rows_by_path: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------
    # STEP 1: Build image manifest
    # ---------------------------------------------------------

    print("\nBuilding image manifest...")

    for path in tqdm(
        image_paths,
        desc="Hashing images",
        unit="image",
    ):

        rel = path.relative_to(dataset_root)

        class_name = (
            rel.parts[0]
            if len(rel.parts) > 1
            else "(root)"
        )

        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            width, height = "", ""

        digest = sha256(path)

        row = {
            "relative_file_path": str(rel).replace("\\", "/"),
            "class_name": class_name,
            "file_name": path.name,
            "extension": path.suffix.lower(),
            "width": width,
            "height": height,
            "file_size_bytes": path.stat().st_size,

            "sha256_hash": digest,

            "exact_duplicate_group_id": "EXD_NONE",
            "near_duplicate_status": "not evaluated",
            "status": "verified",
        }

        image_rows.append(row)

        rows_by_path[
            row["relative_file_path"]
        ] = row

        sha256_to_rows[digest].append(row)

    # ---------------------------------------------------------
    # STEP 2: Exact duplicate groups
    # ---------------------------------------------------------

    print("\nChecking exact duplicates...")

    group_entries: list[dict[str, Any]] = []

    for idx, (_, rows) in enumerate(
        sorted(sha256_to_rows.items(), key=lambda item: item[0]),
        start=1,
    ):

        group_id = f"EXD_{idx:05d}"

        for row in rows:
            row["exact_duplicate_group_id"] = group_id

        if len(rows) > 1:

            group_entries.append(
                {
                    "group_id": group_id,
                    "duplicate_type": "exact_duplicate",
                    "number_of_files": len(rows),
                    "class_information": ", ".join(
                        sorted(
                            {
                                r["class_name"]
                                for r in rows
                            }
                        )
                    ),
                    "member_file_paths": "; ".join(
                        r["relative_file_path"]
                        for r in rows
                    ),
                    "status": "verified",
                }
            )

    print(
        f"Unique content hashes: {len(sha256_to_rows)}"
    )

    print(
        f"Exact duplicate groups: {len(group_entries)}"
    )

    # ---------------------------------------------------------
    # STEP 3: Perceptual hashes
    # ---------------------------------------------------------

    print("\nCalculating perceptual hashes...")

    perceptual_map: dict[str, str] = {}

    for path in tqdm(
        image_paths,
        desc="Computing dHash",
        unit="image",
    ):

        rel = str(
            path.relative_to(dataset_root)
        ).replace("\\", "/")

        try:
            with Image.open(path) as image:
                perceptual_map[rel] = dhash(image)

        except Exception:
            perceptual_map[rel] = ""

    # ---------------------------------------------------------
    # STEP 4: Efficient near-duplicate search
    # ---------------------------------------------------------

    print("\nSearching for near-duplicates...")

    tree = BKTree()

    hash_to_paths: dict[int, list[str]] = defaultdict(list)

    for path, hash_string in perceptual_map.items():

        if not hash_string:
            continue

        hash_int = int(hash_string, 2)

        tree.add(hash_int, path)

        hash_to_paths[hash_int].append(path)

    near_rows: list[dict[str, Any]] = []

    processed_pairs: set[tuple[str, str]] = set()

    sorted_paths = sorted(perceptual_map)

    for left_path in tqdm(
        sorted_paths,
        desc="Comparing perceptual hashes",
        unit="image",
    ):

        left_hash_string = perceptual_map[left_path]

        if not left_hash_string:
            continue

        left_hash = int(left_hash_string, 2)

        matches = tree.search(
            left_hash,
            NEAR_DUPLICATE_THRESHOLD,
        )

        for _, right_path, distance in matches:

            if right_path == left_path:
                continue

            pair = tuple(
                sorted(
                    (
                        left_path,
                        right_path,
                    )
                )
            )

            if pair in processed_pairs:
                continue

            processed_pairs.add(pair)

            left_row = rows_by_path[left_path]
            right_row = rows_by_path[right_path]

            same_class = (
                left_row["class_name"]
                == right_row["class_name"]
            )

            near_rows.append(
                {
                    "pair_id": (
                        f"ND_{len(near_rows) + 1:05d}"
                    ),
                    "file_a": pair[0],
                    "file_b": pair[1],
                    "distance": distance,
                    "same_class": str(same_class),
                    "review_status": (
                        "candidate_near_duplicate"
                    ),
                    "status": "verified",
                }
            )

    # ---------------------------------------------------------
    # STEP 5: Write outputs
    # ---------------------------------------------------------

    print("\nWriting manifest files...")

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        report_dir / "image_manifest.csv",
        [
            "relative_file_path",
            "class_name",
            "file_name",
            "extension",
            "width",
            "height",
            "file_size_bytes",
            "sha256_hash",
            "exact_duplicate_group_id",
            "near_duplicate_status",
            "status",
        ],
        image_rows,
    )

    write_csv(
        report_dir / "duplicate_groups.csv",
        [
            "group_id",
            "duplicate_type",
            "number_of_files",
            "class_information",
            "member_file_paths",
            "status",
        ],
        group_entries,
    )

    write_csv(
        report_dir / "near_duplicate_review.csv",
        [
            "pair_id",
            "file_a",
            "file_b",
            "distance",
            "same_class",
            "review_status",
            "status",
        ],
        near_rows,
    )

    manifest_report = {
        "status": "verified",
        "dataset_root": str(dataset_root),
        "total_images": len(image_rows),
        "unique_content_hashes": len(sha256_to_rows),
        "exact_duplicate_groups": len(group_entries),
        "images_in_exact_duplicate_groups": sum(
            item["number_of_files"]
            for item in group_entries
        ),
        "near_duplicate_candidate_pairs": len(
            near_rows
        ),
        "policy": {
            "exact_duplicates_grouped_by_sha256": True,
            "raw_images_left_intact": True,
            "near_duplicates_not_auto_deleted": True,
            "cross_class_near_duplicates_preserved": True,
        },
    }

    (
        report_dir / "manifest_status.json"
    ).write_text(
        json.dumps(
            manifest_report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nManifest generation complete.")
    print(f"Total images: {len(image_rows)}")
    print(
        f"Unique content hashes: "
        f"{len(sha256_to_rows)}"
    )
    print(
        f"Exact duplicate groups: "
        f"{len(group_entries)}"
    )
    print(
        f"Near-duplicate pairs: "
        f"{len(near_rows)}"
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
    )

    args = parser.parse_args()

    try:
        build_manifest(args.config)

    except FileNotFoundError as exc:

        config = load_config(args.config)

        report_dir = Path(
            config.get("data", {}).get(
                "dataset_report_dir",
                "data/dataset_report",
            )
        )

        blocked_manifest(
            report_dir,
            str(
                config.get(
                    "data",
                    {}
                ).get(
                    "external_dataset_path",
                    "",
                )
            ),
            str(exc),
        )

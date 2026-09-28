"""Dataset and split loading for the verified Phase 2 CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from PIL import Image
from torch.utils.data import Dataset


EXPECTED_COLUMNS = ("image_path", "class_name")
EXPECTED_SPLIT_SIZES = {"train": 12710, "validation": 2725, "test": 2725}


def read_split_csv(split_path: str | Path) -> list[dict[str, str]]:
    path = Path(split_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(f"{path} must contain columns {EXPECTED_COLUMNS}.")
        rows = list(reader)
    if not rows or any(not row["image_path"] or not row["class_name"] for row in rows):
        raise ValueError(f"{path} contains no rows or incomplete rows.")
    return rows


def load_phase2_splits(splits_dir: str | Path) -> dict[str, list[dict[str, str]]]:
    directory = Path(splits_dir)
    splits = {
        name: read_split_csv(directory / f"{name}.csv")
        for name in ("train", "validation", "test")
    }
    for name, expected_size in EXPECTED_SPLIT_SIZES.items():
        if len(splits[name]) != expected_size:
            raise ValueError(
                f"{name} split has {len(splits[name])} rows; expected {expected_size}."
            )
    train_classes = sorted({row["class_name"] for row in splits["train"]})
    if len(train_classes) != 10:
        raise ValueError(f"Expected 10 classes in train split, found {len(train_classes)}.")
    if any(row["class_name"] not in train_classes for rows in splits.values() for row in rows):
        raise ValueError("Validation/test split contains a class absent from train.")
    return splits


class TomatoLeafDataset(Dataset):
    """Loads paths from one immutable Phase 2 split without changing row order."""

    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        dataset_root: str | Path,
        class_to_index: dict[str, int],
        transform=None,
    ) -> None:
        self.rows = list(rows)
        self.dataset_root = Path(dataset_root)
        self.class_to_index = dict(class_to_index)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.dataset_root / row["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image referenced by split is missing: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_index[row["class_name"]]

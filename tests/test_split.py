"""Tests for the real group-aware train/val/test splits."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.split import split_dataset

pytestmark = pytest.mark.slow

SPLIT_NAMES = ("train", "val", "test")


def _md5(path: Path) -> str:
    """Return the MD5 hex digest of a file."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_real() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load dataset.csv and the three split CSVs from config paths."""
    cfg = load_config()
    dataset = pd.read_csv(Path(cfg.paths.processed_dir) / "dataset.csv")
    splits = {
        name: pd.read_csv(Path(cfg.paths.splits_dir) / f"{name}.csv") for name in SPLIT_NAMES
    }
    return dataset, splits


def test_no_group_id_in_more_than_one_split() -> None:
    """No group_id appears in more than one split."""
    _, splits = _load_real()
    groups = {name: set(frame["group_id"]) for name, frame in splits.items()}
    assert not (groups["train"] & groups["val"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["val"] & groups["test"])


def test_no_identical_text_clean_across_splits() -> None:
    """No identical text_clean appears in more than one split."""
    _, splits = _load_real()
    texts = {name: set(frame["text_clean"].astype(str)) for name, frame in splits.items()}
    assert not (texts["train"] & texts["val"])
    assert not (texts["train"] & texts["test"])
    assert not (texts["val"] & texts["test"])


def test_split_sizes_within_two_points_of_targets() -> None:
    """Split sizes are within +/- 2 percentage points of 70/15/15."""
    cfg = load_config()
    dataset, splits = _load_real()
    n = len(dataset)
    targets = {"train": float(cfg.split.train), "val": float(cfg.split.val), "test": float(cfg.split.test)}
    for name, target in targets.items():
        share = len(splits[name]) / n
        assert abs(share - target) <= 0.02, f"{name} share {share:.4f} vs target {target:.2f}"


def test_class_share_within_three_points_of_overall() -> None:
    """Each class's share inside each split is within +/- 3 points of its overall share."""
    dataset, splits = _load_real()
    overall = dataset["label"].value_counts(normalize=True)
    for name, frame in splits.items():
        inside = frame["label"].value_counts(normalize=True)
        for label, overall_share in overall.items():
            split_share = float(inside.get(label, 0.0))
            assert abs(split_share - float(overall_share)) <= 0.03, (
                f"{name} label {label}: {split_share:.4f} vs overall {float(overall_share):.4f}"
            )


def test_row_count_matches_dataset_and_ids_are_disjoint() -> None:
    """Train + val + test equals dataset.csv length, and ids are disjoint."""
    dataset, splits = _load_real()
    assert sum(len(frame) for frame in splits.values()) == len(dataset)
    ids = {name: set(frame["id"]) for name, frame in splits.items()}
    assert not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])
    assert ids["train"] | ids["val"] | ids["test"] == set(dataset["id"])


def test_two_runs_write_identical_split_files() -> None:
    """Running the splitter twice writes byte-identical train/val/test CSVs."""
    cfg = load_config()
    dataset = pd.read_csv(Path(cfg.paths.processed_dir) / "dataset.csv")
    split_dir = Path(cfg.paths.splits_dir)
    split_dataset(dataset, cfg)
    first = {name: _md5(split_dir / f"{name}.csv") for name in SPLIT_NAMES}
    split_dataset(dataset, cfg)
    second = {name: _md5(split_dir / f"{name}.csv") for name in SPLIT_NAMES}
    assert first == second

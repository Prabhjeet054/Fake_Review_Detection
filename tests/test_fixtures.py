"""Checks that the shared tmp fixtures stay off the real data paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import AttrDict, find_project_root


def test_tiny_reviews_cover_all_three_classes(tiny_reviews_df: pd.DataFrame) -> None:
    """The shared corpus has 30 rows and 10 examples of each class."""
    assert len(tiny_reviews_df) == 30
    counts = tiny_reviews_df["label"].value_counts().to_dict()
    assert counts == {0: 10, 1: 10, 2: 10}


def test_tmp_config_does_not_point_at_real_data(tmp_config: AttrDict, tmp_path: Path) -> None:
    """Temporary config paths stay under tmp_path, not the project data/outputs dirs."""
    root = find_project_root()
    assert str(tmp_config.paths.raw_dir).startswith(str(tmp_path))
    assert str(tmp_config.paths.splits_dir).startswith(str(tmp_path))
    assert str(tmp_config.paths.metrics_dir).startswith(str(tmp_path))
    assert root / "data" / "raw" != tmp_config.paths.raw_dir
    assert (tmp_config.paths.raw_dir / "fake_reviews_dataset.csv").is_file()
    assert (tmp_config.paths.raw_dir / "deceptive-opinion.csv").is_file()

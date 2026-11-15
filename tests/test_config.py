"""Tests for src.config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import find_project_root, load_config, to_attr_dict, validate_config


def test_load_config_attribute_access() -> None:
    """Loaded config supports nested attribute access."""
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.model.name == "distilbert-base-uncased"
    assert cfg.split.train + cfg.split.val + cfg.split.test == pytest.approx(1.0)


def test_paths_are_absolute_and_independent_of_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Path values resolve against the project root even if cwd changes."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    root = find_project_root()
    assert cfg.paths.raw_dir.is_absolute()
    assert cfg.paths.raw_dir == (root / "data" / "raw").resolve()
    assert cfg.data.salminen_file == (root / "data" / "raw" / "fake_reviews_dataset.csv").resolve()
    assert cfg.data.ott_file == (root / "data" / "raw" / "deceptive-opinion.csv").resolve()
    assert cfg.data.min_words == 5


def test_missing_section_raises(tmp_path: Path) -> None:
    """A config missing a required section raises ValueError."""
    payload = yaml.safe_load((find_project_root() / "config.yaml").read_text(encoding="utf-8"))
    del payload["split"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required config section"):
        load_config(path)


def test_split_sum_validation() -> None:
    """Split ratios that do not sum to 1.0 raise ValueError."""
    cfg = to_attr_dict(
        {
            "seed": 42,
            "paths": {},
            "data": {},
            "labels": {0: "a", 1: "b", 2: "c"},
            "split": {"train": 0.5, "val": 0.2, "test": 0.2},
            "dedup": {},
            "tfidf": {},
            "model": {"num_classes": 3},
            "training": {},
        }
    )
    with pytest.raises(ValueError, match="split ratios must sum to 1.0"):
        validate_config(cfg)


def test_num_classes_must_match_labels() -> None:
    """model.num_classes must equal the number of labels."""
    cfg = to_attr_dict(
        {
            "seed": 42,
            "paths": {},
            "data": {},
            "labels": {0: "a", 1: "b", 2: "c"},
            "split": {"train": 0.7, "val": 0.15, "test": 0.15},
            "dedup": {},
            "tfidf": {},
            "model": {"num_classes": 2},
            "training": {},
        }
    )
    with pytest.raises(ValueError, match="model.num_classes"):
        validate_config(cfg)

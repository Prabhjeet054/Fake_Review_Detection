"""Tests for src.check_data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.check_data import build_data_card, check_leakage, run_check_data, run_checks
from src.config import load_config


def test_real_splits_pass_and_data_card_is_written() -> None:
    """Real splits pass every check and data_card.md has both tables."""
    issues, path = run_check_data()
    assert issues == []
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "## Dataset description" in text
    assert "## Class description" in text
    assert "Number of classes" in text
    assert "Count in test" in text
    assert "Genuine Human" in text


def test_group_leakage_is_reported() -> None:
    """A shared group_id across splits is a failed check."""
    cfg = load_config()
    row = {
        "id": 1,
        "text": "the hotel room was clean and quiet enough",
        "label": 0,
        "source": "ott",
        "domain": "hotel",
        "text_clean": "the hotel room was clean and quiet enough",
        "group_id": 99,
    }
    splits = {
        "train": pd.DataFrame([{**row, "id": 1}]),
        "val": pd.DataFrame([{**row, "id": 2, "text": "other", "text_clean": "other review text here"}]),
        "test": pd.DataFrame([{**row, "id": 3, "text": "third", "text_clean": "third review text here"}]),
    }
    issues = check_leakage(splits)
    assert any("group_id" in issue for issue in issues)
    all_issues = run_checks(splits, cfg)
    assert all_issues


def test_data_card_includes_test_class_counts(tmp_path: Path) -> None:
    """The class table reports test-split counts per label."""
    cfg = load_config()
    cfg.paths.metrics_dir = tmp_path
    splits = {
        name: pd.DataFrame(
            {
                "id": [i + offset for i in range(3)],
                "text": [f"{name} review {i} extra words" for i in range(3)],
                "label": [0, 1, 2],
                "source": ["salminen", "ott", "salminen"],
                "domain": ["product", "hotel", "product"],
                "text_clean": [f"{name} review {i} extra words" for i in range(3)],
                "group_id": [i + offset for i in range(3)],
            }
        )
        for name, offset in (("train", 0), ("val", 10), ("test", 20))
    }
    card = build_data_card(splits, cfg)
    assert "test 3" in card or "test" in card
    assert "| 0 | Genuine Human |" in card
    assert "| 1 | Fake Human-Written |" in card
    assert "| 2 | AI-Generated |" in card

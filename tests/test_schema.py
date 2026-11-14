"""Tests for src.schema."""

import pandas as pd
import pytest

from src.schema import add_label_names, allowed_labels, required_columns, validate_reviews


def test_required_columns_and_labels(cfg: dict) -> None:
    """Schema helpers read from config."""
    assert "review_id" in required_columns(cfg)
    assert allowed_labels(cfg) == {0, 1, 2}


def test_validate_reviews_rejects_bad_label(cfg: dict, reviews_df: pd.DataFrame) -> None:
    """Invalid class ids raise ValueError."""
    bad = reviews_df.copy()
    bad.loc[0, cfg["columns"]["label"]] = 9
    with pytest.raises(ValueError, match="Invalid labels"):
        validate_reviews(bad, cfg)


def test_validate_reviews_adds_names(cfg: dict, reviews_df: pd.DataFrame) -> None:
    """Label names are attached when missing."""
    frame = reviews_df.drop(columns=[cfg["columns"]["label_name"]])
    out = validate_reviews(frame, cfg)
    assert cfg["columns"]["label_name"] in out.columns
    named = add_label_names(frame, cfg)
    assert named[cfg["columns"]["label_name"]].notna().all()

"""Tests for src.analysis."""

import numpy as np
import pandas as pd

from src.analysis import attach_predictions, confusion_pairs, error_slice


def test_error_slice_and_pairs(cfg: dict, reviews_df: pd.DataFrame) -> None:
    """Misclassified rows are isolated and confusion pairs counted."""
    subset = reviews_df.head(6).copy()
    ids = subset[cfg["columns"]["id"]].astype(str).tolist()
    y_true = subset[cfg["columns"]["label"]].astype(int).to_numpy()
    y_pred = y_true.copy()
    y_pred[0] = (y_pred[0] + 1) % 3
    merged = attach_predictions(subset, ids, y_true, y_pred, cfg)
    assert merged["is_error"].sum() == 1
    errors = error_slice(merged, cfg)
    assert len(errors) == 1
    pairs = confusion_pairs(merged)
    assert pairs[0]["count"] == 1

"""Tests for src.evaluate."""

import numpy as np

from src.evaluate import compute_metrics, grouped_binary_metrics


def test_perfect_predictions(cfg: dict) -> None:
    """Perfect labels yield accuracy 1 and kappa 1."""
    y = np.array([0, 1, 2, 0, 1, 2])
    metrics = compute_metrics(y, y, cfg)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert len(metrics["per_class"]) == 3


def test_grouped_binary(cfg: dict) -> None:
    """Genuine vs fake grouping treats 1 and 2 as the positive class."""
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2])
    grouped = grouped_binary_metrics(y_true, y_pred, cfg)
    assert "genuine_vs_fake" in grouped
    assert grouped["genuine_vs_fake"]["support"] == 4

"""Tests for src.plots."""

import numpy as np

from src.clean import clean_dataframe
from src.plots import plot_class_distribution, plot_confusion_matrix, plot_metrics_comparison


def test_class_distribution_and_confusion(cfg: dict, reviews_df) -> None:
    """Plot helpers write image files under the configured figures directory."""
    cleaned = clean_dataframe(reviews_df, cfg)
    path = plot_class_distribution(cleaned, cfg, "test_class_distribution")
    assert path.exists()
    matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    cm_path = plot_confusion_matrix(matrix, cfg, "test_cm")
    assert cm_path.exists()
    rows = [{"model": "knn", "macro_f1": 0.5}, {"model": "svm", "macro_f1": 0.6}]
    cmp_path = plot_metrics_comparison(rows, cfg, "test_cmp")
    assert cmp_path.exists()

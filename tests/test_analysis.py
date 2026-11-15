"""Tests for confusion-matrix analysis."""

import numpy as np

from src.analysis import binary_metrics, error_breakdown

CONFUSION_MATRIX = np.array(
    [
        [79, 4, 3],
        [60, 242, 11],
        [7, 43, 135],
    ]
)


def test_binary_metrics_known_confusion_matrix() -> None:
    """Grouped metrics match the fixed known confusion matrix."""
    result = binary_metrics(CONFUSION_MATRIX)
    overall_accuracy = np.trace(CONFUSION_MATRIX) / CONFUSION_MATRIX.sum()

    assert round(float(overall_accuracy), 4) == 0.7808
    assert {
        key: round(value, 4)
        for key, value in result["genuine_vs_deceptive"].items()
    } == {
        "accuracy": 0.8733,
        "precision": 0.9840,
        "recall": 0.8655,
    }
    assert {
        key: round(value, 4)
        for key, value in result["human_vs_ai_generated"].items()
    } == {
        "accuracy": 0.8904,
        "precision": 0.9060,
        "recall": 0.7297,
    }


def test_error_breakdown_known_confusion_matrix() -> None:
    """Error counts and shares are sorted correctly for the fixed matrix."""
    result = error_breakdown(CONFUSION_MATRIX)

    assert int(result["count"].sum()) == 128
    assert result.iloc[0]["true_class"] == 1
    assert result.iloc[0]["predicted_class"] == 0
    assert result.iloc[0]["count"] == 60
    assert round(float(result.iloc[0]["share_of_errors"]), 4) == 0.4688
    assert result.iloc[1]["true_class"] == 2
    assert result.iloc[1]["predicted_class"] == 1
    assert result.iloc[1]["count"] == 43

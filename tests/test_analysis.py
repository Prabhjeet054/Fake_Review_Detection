"""Tests for confusion-matrix analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import binary_metrics, error_analysis, error_breakdown

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


def test_error_analysis_writes_misclassified_rows_and_report(tmp_path: Path) -> None:
    """Detailed error analysis writes names, confidence, domain, and length results."""
    metrics_dir = tmp_path / "metrics"
    splits_dir = tmp_path / "splits"
    metrics_dir.mkdir()
    splits_dir.mkdir()
    pd.DataFrame(
        [
            {"id": 1, "text_clean": " ".join(["hotel"] * 49), "true": 0, "pred": 1, "p0": 0.1, "p1": 0.8, "p2": 0.1},
            {"id": 2, "text_clean": " ".join(["product"] * 120), "true": 2, "pred": 0, "p0": 0.7, "p1": 0.1, "p2": 0.2},
            {"id": 3, "text_clean": " ".join(["hotel"] * 60), "true": 1, "pred": 1, "p0": 0.1, "p1": 0.8, "p2": 0.1},
        ]
    ).to_csv(metrics_dir / "test_predictions.csv", index=False)
    pd.DataFrame({"id": [1, 2, 3], "domain": ["hotel", "product", "hotel"]}).to_csv(
        splits_dir / "test.csv", index=False
    )
    cfg = {
        "paths": {"metrics_dir": metrics_dir, "splits_dir": splits_dir},
        "labels": {0: "Genuine Human", 1: "Fake Human-Written", 2: "AI-Generated"},
    }

    result = error_analysis(cfg)

    misclassified = pd.read_csv(result["misclassified_path"])
    assert len(misclassified) == 2
    assert list(misclassified["predicted_class"]) == ["Fake Human-Written", "Genuine Human"]
    assert list(misclassified["predicted_probability"]) == [0.8, 0.7]
    report = result["report_path"].read_text(encoding="utf-8")
    assert "Error rate by domain" in report
    assert "Error rate by review length" in report
    assert "Genuine Human -> Fake Human-Written" in report


@pytest.mark.slow
def test_real_misclassified_output_matches_metrics() -> None:
    """The real misclassified artifact has the expected rows and labels."""
    root = Path(__file__).resolve().parent.parent
    metrics_dir = root / "outputs" / "metrics"
    metrics = json.loads((metrics_dir / "metrics.json").read_text(encoding="utf-8"))
    errors = pd.read_csv(metrics_dir / "misclassified.csv")

    assert len(errors) == metrics["n_test"] - metrics["n_correct"]
    assert (errors["true_class"] != errors["predicted_class"]).all()
    assert errors["predicted_probability"].is_monotonic_decreasing


@pytest.mark.slow
def test_real_length_buckets_cover_test_rows_once() -> None:
    """The markdown length table accounts for every test row exactly once."""
    root = Path(__file__).resolve().parent.parent
    report = (root / "outputs" / "metrics" / "error_analysis.md").read_text(encoding="utf-8")
    metrics = json.loads(
        (root / "outputs" / "metrics" / "metrics.json").read_text(encoding="utf-8")
    )
    buckets = ("<50", "50-100", "100-200", ">200")
    totals = []
    for bucket in buckets:
        row = next(line for line in report.splitlines() if line.startswith(f"| {bucket} |"))
        totals.append(int(row.split("|")[2].strip()))
    assert sum(totals) == metrics["n_test"]

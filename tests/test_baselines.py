"""Tests for saved baseline metrics and test-split predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.config import load_config

pytestmark = pytest.mark.slow

METRIC_COLUMNS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
EXPECTED_MODELS = {
    "Majority class",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "AdaBoost",
    "Gradient Boosting",
}


def _metrics_paths() -> tuple[Path, Path]:
    """Return paths to baselines.csv and the test prediction file."""
    cfg = load_config()
    metrics_dir = Path(cfg.paths.metrics_dir)
    return metrics_dir / "baselines.csv", metrics_dir / "baseline_predictions_test.csv"


def test_baselines_csv_has_fourteen_rows_and_metrics_in_unit_interval() -> None:
    """baselines.csv has 7 models x 2 splits and every metric is in [0, 1]."""
    metrics_path, _ = _metrics_paths()
    df = pd.read_csv(metrics_path)
    assert len(df) == 14
    assert set(df["model"]) == EXPECTED_MODELS
    assert set(df["split"]) == {"val", "test"}
    assert df.groupby("model").size().eq(2).all()
    for column in METRIC_COLUMNS:
        assert df[column].between(0.0, 1.0).all(), column


def test_recomputed_svm_test_metrics_match_csv() -> None:
    """Accuracy and macro F1 recomputed from test predictions match baselines.csv."""
    metrics_path, preds_path = _metrics_paths()
    metrics = pd.read_csv(metrics_path)
    preds = pd.read_csv(preds_path)
    row = metrics.loc[(metrics["model"] == "SVM") & (metrics["split"] == "test")].iloc[0]
    y_true = preds["y_true"].to_numpy()
    y_pred = preds["svm"].to_numpy()
    assert accuracy_score(y_true, y_pred) == row["accuracy"]
    assert f1_score(y_true, y_pred, average="macro", zero_division=0) == row["macro_f1"]


def test_every_real_model_beats_majority_class_test_accuracy() -> None:
    """Every trained model's test accuracy is higher than the majority-class row."""
    metrics_path, _ = _metrics_paths()
    df = pd.read_csv(metrics_path)
    test = df.loc[df["split"] == "test"].set_index("model")
    majority = float(test.loc["Majority class", "accuracy"])
    for name in EXPECTED_MODELS - {"Majority class"}:
        assert float(test.loc[name, "accuracy"]) > majority, name

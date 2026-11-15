"""Tests that metrics.json matches hand-computed confusion-matrix formulas."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import classification_report

from src.config import find_project_root, load_config
from src.evaluate import class_names, evaluate_test
from src.utils import load_json

pytestmark = pytest.mark.slow

METRICS_PATH = find_project_root() / "outputs" / "metrics" / "metrics.json"
PREDS_PATH = find_project_root() / "outputs" / "metrics" / "test_predictions.csv"
TOL = 1e-9


def _require_eval_outputs() -> None:
    """Fail clearly if the real evaluate outputs are missing."""
    if not METRICS_PATH.is_file() or not PREDS_PATH.is_file():
        raise FileNotFoundError("Missing metrics.json or test_predictions.csv. Run python -m src.evaluate first.")


def _metrics_from_confusion(cm: np.ndarray) -> dict[str, Any]:
    """Recompute accuracy, per-class P/R/F1, averages, and kappa from a confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    n_test = float(cm.sum())
    support = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    tp = np.diag(cm)
    precision = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1_den = precision + recall
    f1 = np.divide(2.0 * precision * recall, f1_den, out=np.zeros_like(tp), where=f1_den != 0)
    accuracy = float(tp.sum() / n_test) if n_test else 0.0
    p_e = float((support / n_test) @ (col_sum / n_test)) if n_test else 0.0
    kappa = float((accuracy - p_e) / (1.0 - p_e)) if p_e != 1.0 else 1.0
    return {
        "n_test": int(cm.sum()),
        "n_correct": int(tp.sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float(np.average(precision, weights=support)),
        "weighted_recall": float(np.average(recall, weights=support)),
        "weighted_f1": float(np.average(f1, weights=support)),
        "cohen_kappa": kappa,
    }


def test_confusion_matrix_totals_and_row_sums_match_support() -> None:
    """Matrix sum equals n_test; each row sum equals that class's support."""
    _require_eval_outputs()
    metrics = load_json(METRICS_PATH)
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    names = class_names(load_config())
    assert int(cm.sum()) == int(metrics["n_test"])
    row_sums = cm.sum(axis=1)
    for index, name in enumerate(names):
        assert int(row_sums[index]) == int(metrics["per_class"][name]["support"])


def test_hand_metrics_from_confusion_matrix_match_json() -> None:
    """Accuracy, P/R/F1, averages, and kappa from the matrix match metrics.json within 1e-9."""
    _require_eval_outputs()
    metrics = load_json(METRICS_PATH)
    cm = np.asarray(metrics["confusion_matrix"])
    hand = _metrics_from_confusion(cm)
    names = class_names(load_config())
    assert hand["n_test"] == metrics["n_test"]
    assert hand["n_correct"] == metrics["n_correct"]
    assert hand["accuracy"] == pytest.approx(metrics["accuracy"], abs=TOL)
    for index, name in enumerate(names):
        row = metrics["per_class"][name]
        assert hand["precision"][index] == pytest.approx(row["precision"], abs=TOL)
        assert hand["recall"][index] == pytest.approx(row["recall"], abs=TOL)
        assert hand["f1"][index] == pytest.approx(row["f1"], abs=TOL)
        assert int(hand["support"][index]) == int(row["support"])
    assert hand["macro_precision"] == pytest.approx(metrics["macro_precision"], abs=TOL)
    assert hand["macro_recall"] == pytest.approx(metrics["macro_recall"], abs=TOL)
    assert hand["macro_f1"] == pytest.approx(metrics["macro_f1"], abs=TOL)
    assert hand["weighted_precision"] == pytest.approx(metrics["weighted_precision"], abs=TOL)
    assert hand["weighted_recall"] == pytest.approx(metrics["weighted_recall"], abs=TOL)
    assert hand["weighted_f1"] == pytest.approx(metrics["weighted_f1"], abs=TOL)
    assert hand["cohen_kappa"] == pytest.approx(metrics["cohen_kappa"], abs=TOL)
    report = classification_report(
        pd.read_csv(PREDS_PATH)["true"],
        pd.read_csv(PREDS_PATH)["pred"],
        labels=list(range(len(names))),
        target_names=names,
        zero_division=0,
        digits=4,
    )
    print(report)


def test_pred_is_argmax_and_probabilities_sum_to_one() -> None:
    """Every row: pred = argmax(p0, p1, p2) and p0+p1+p2 = 1 within 1e-5."""
    _require_eval_outputs()
    frame = pd.read_csv(PREDS_PATH)
    probs = frame[["p0", "p1", "p2"]].to_numpy(dtype=np.float64)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(frame["pred"].to_numpy(), probs.argmax(axis=1))


def test_rerunning_evaluate_writes_identical_metrics_json() -> None:
    """A second python -m src.evaluate run writes the same metrics.json."""
    _require_eval_outputs()
    before = METRICS_PATH.read_text(encoding="utf-8")
    evaluate_test(load_config())
    after = METRICS_PATH.read_text(encoding="utf-8")
    assert after == before

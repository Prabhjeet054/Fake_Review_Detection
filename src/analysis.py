"""Confusion-matrix metrics and error analysis for the held-out test split."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

CLASS_NAMES = ("Genuine Human", "Fake Human-Written", "AI-Generated")


def _validate_confusion_matrix(cm: Any) -> np.ndarray:
    """Return a validated 3x3 confusion matrix as floating-point counts."""
    matrix = np.asarray(cm)
    if matrix.shape != (3, 3):
        raise ValueError(f"cm must have shape (3, 3), got {matrix.shape}")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("cm must contain numeric counts")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("cm must contain finite, non-negative counts")
    if not np.equal(matrix, np.floor(matrix)).all():
        raise ValueError("cm must contain integer counts")
    return matrix.astype(np.float64, copy=False)


def _binary_scores(matrix: np.ndarray, positive: tuple[int, ...]) -> dict[str, float]:
    """Calculate accuracy, precision, and recall for a grouped binary task."""
    positive_mask = np.zeros(3, dtype=bool)
    positive_mask[list(positive)] = True
    true_positive = float(matrix[np.ix_(positive_mask, positive_mask)].sum())
    false_positive = float(matrix[~positive_mask][:, positive_mask].sum())
    false_negative = float(matrix[positive_mask][:, ~positive_mask].sum())
    true_negative = float(matrix[~positive_mask][:, ~positive_mask].sum())
    total = true_positive + false_positive + false_negative + true_negative
    return {
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        ),
        "recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        ),
    }


def binary_metrics(cm: Any) -> dict[str, dict[str, float]]:
    """Return grouped binary metrics for the two requested class mappings."""
    matrix = _validate_confusion_matrix(cm)
    return {
        "genuine_vs_deceptive": _binary_scores(matrix, (1, 2)),
        "human_vs_ai_generated": _binary_scores(matrix, (2,)),
    }


def error_breakdown(cm: Any) -> pd.DataFrame:
    """Return every off-diagonal confusion cell and its share of all errors."""
    matrix = _validate_confusion_matrix(cm)
    total_errors = float(matrix.sum() - np.trace(matrix))
    rows = [
        {
            "true_class": true_class,
            "predicted_class": predicted_class,
            "count": int(matrix[true_class, predicted_class]),
            "share_of_errors": (
                float(matrix[true_class, predicted_class] / total_errors)
                if total_errors
                else 0.0
            ),
        }
        for true_class in range(3)
        for predicted_class in range(3)
        if true_class != predicted_class
    ]
    return pd.DataFrame(rows).sort_values(
        ["count", "true_class", "predicted_class"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def attach_predictions(
    df: pd.DataFrame,
    ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Join model predictions onto the original test frame."""
    id_col = cfg["columns"]["id"]
    pred_df = pd.DataFrame(
        {
            id_col: ids,
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
        }
    )
    merged = df.merge(pred_df, on=id_col, how="inner")
    names = cfg.get("labels", {index: name for index, name in enumerate(CLASS_NAMES)})
    names = {int(key): str(value) for key, value in names.items()}
    merged["true_name"] = merged["y_true"].map(names)
    merged["pred_name"] = merged["y_pred"].map(names)
    merged["is_error"] = merged["y_true"] != merged["y_pred"]
    return merged


def error_slice(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return misclassified rows with useful debugging columns."""
    text_col = cfg["columns"]["clean_text"] if cfg["columns"]["clean_text"] in df.columns else cfg["columns"]["text"]
    keep = [
        cfg["columns"]["id"],
        cfg["columns"]["source"],
        text_col,
        "true_name",
        "pred_name",
        "y_true",
        "y_pred",
    ]
    keep = [col for col in keep if col in df.columns]
    errors = df.loc[df["is_error"], keep].copy()
    errors["n_tokens"] = df.loc[df["is_error"], text_col].fillna("").astype(str).str.split().map(len)
    return errors.reset_index(drop=True)


def confusion_pairs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Count each (true, predicted) error pair."""
    errors = df.loc[df["is_error"]]
    if errors.empty:
        return []
    counts = errors.groupby(["true_name", "pred_name"]).size().reset_index(name="count")
    counts = counts.sort_values("count", ascending=False)
    return counts.to_dict(orient="records")


def metrics_by_source(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute overall accuracy per data source."""
    from src.evaluate import compute_metrics

    source_col = cfg["columns"]["source"]
    if source_col not in df.columns:
        return {}
    out: dict[str, Any] = {}
    for source, group in df.groupby(source_col):
        out[str(source)] = compute_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy(), cfg)
    return out


def main(argv: list[str] | None = None) -> None:
    """Write grouped binary metrics and confusion-cell errors from metrics.json."""
    from src.config import load_config
    from src.utils import load_json, save_json

    parser = argparse.ArgumentParser(description="Analyze the saved test confusion matrix.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    metrics_dir = cfg.paths.metrics_dir
    metrics_path = metrics_dir / "metrics.json"
    metrics = load_json(metrics_path)
    matrix = metrics["confusion_matrix"]
    binary_path = save_json(binary_metrics(matrix), metrics_dir / "binary_metrics.json")
    errors_path = metrics_dir / "error_breakdown.csv"
    error_breakdown(matrix).to_csv(errors_path, index=False)
    print(f"wrote {binary_path}")
    print(f"wrote {errors_path}")


if __name__ == "__main__":
    main()

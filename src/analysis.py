"""Confusion-matrix metrics and error analysis for the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path
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


def _config_path(cfg: Any, section: str, key: str) -> Path:
    """Return a path from either mapping-style or attribute-style config."""
    values = cfg[section] if isinstance(cfg, dict) else getattr(cfg, section)
    value = values[key] if isinstance(values, dict) else getattr(values, key)
    return Path(value)


def _label_map(cfg: Any) -> dict[int, str]:
    """Return configured class names, falling back to the project labels."""
    labels = cfg.get("labels") if isinstance(cfg, dict) else getattr(cfg, "labels", None)
    if labels is None:
        return dict(enumerate(CLASS_NAMES))
    return {int(key): str(value) for key, value in labels.items()}


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as a readable markdown table."""
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            text = str(value).replace("|", "\\|").replace("\n", " ")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _length_bucket(word_count: int) -> str:
    """Return the configured review-length bucket for a word count."""
    if word_count < 50:
        return "<50"
    if word_count <= 100:
        return "50-100"
    if word_count <= 200:
        return "100-200"
    return ">200"


def _group_error_rates(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calculate total rows, errors, and error rate for a grouping column."""
    grouped = (
        frame.groupby(column, dropna=False, observed=False)
        .agg(total=("is_error", "size"), errors=("is_error", "sum"))
        .reset_index()
    )
    grouped["error_rate"] = grouped["errors"] / grouped["total"]
    grouped["error_rate"] = grouped["error_rate"].map(lambda value: f"{value:.4f}")
    return grouped


def error_analysis(cfg: Any | None = None) -> dict[str, Any]:
    """Write misclassified rows and a markdown report from test predictions."""
    if cfg is None:
        from src.config import load_config

        cfg = load_config()
    metrics_dir = _config_path(cfg, "paths", "metrics_dir")
    predictions_path = metrics_dir / "test_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    required = {"id", "text_clean", "true", "pred", "p0", "p1", "p2"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"test_predictions.csv is missing column(s): {', '.join(missing)}")

    labels = _label_map(cfg)
    predictions["true"] = predictions["true"].astype(int)
    predictions["pred"] = predictions["pred"].astype(int)
    predictions["is_error"] = predictions["true"] != predictions["pred"]
    probabilities = predictions[["p0", "p1", "p2"]].to_numpy(dtype=float)
    predicted_labels = predictions["pred"].to_numpy(dtype=int)
    predictions["predicted_probability"] = np.take_along_axis(
        probabilities, predicted_labels[:, None], axis=1
    ).ravel()

    split_path = _config_path(cfg, "paths", "splits_dir") / "test.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing test split for domain analysis: {split_path}")
    domains = pd.read_csv(split_path, usecols=["id", "domain"])
    domain_ids = domains["id"].astype(str)
    domains = domains.assign(_join_id=domain_ids).drop(columns=["id"])
    predictions["_join_id"] = predictions["id"].astype(str)
    predictions = predictions.merge(domains, on="_join_id", how="left", validate="one_to_one")
    if predictions["domain"].isna().any():
        missing_ids = predictions.loc[predictions["domain"].isna(), "id"].tolist()
        raise ValueError(f"test split is missing domain for prediction id {missing_ids[0]!r}")
    predictions["word_count"] = predictions["text_clean"].fillna("").astype(str).str.split().str.len()
    predictions["length_bucket"] = predictions["word_count"].map(_length_bucket)

    misclassified = predictions.loc[predictions["is_error"]].copy()
    misclassified["true_class"] = misclassified["true"].map(labels)
    misclassified["predicted_class"] = misclassified["pred"].map(labels)
    misclassified = misclassified.sort_values(
        ["predicted_probability", "true", "pred"],
        ascending=[False, True, True],
    )
    misclassified_path = metrics_dir / "misclassified.csv"
    misclassified[
        [
            "id",
            "text_clean",
            "true_class",
            "predicted_class",
            "predicted_probability",
        ]
    ].to_csv(misclassified_path, index=False)

    domain_rates = _group_error_rates(predictions, "domain")
    length_rates = _group_error_rates(
        predictions.assign(length_bucket=pd.Categorical(
            predictions["length_bucket"],
            categories=["<50", "50-100", "100-200", ">200"],
            ordered=True,
        )),
        "length_bucket",
    )
    pair_tables: dict[str, pd.DataFrame] = {}
    for (true_label, predicted_label), group in misclassified.groupby(
        ["true", "pred"], sort=True
    ):
        pair_name = f"{labels[int(true_label)]} -> {labels[int(predicted_label)]}"
        pair_tables[pair_name] = group.nlargest(5, "predicted_probability")[
            ["id", "text_clean", "predicted_probability"]
        ].rename(columns={"text_clean": "review"})

    report_lines = [
        "# Error Analysis",
        "",
        f"Total predictions: {len(predictions)}",
        f"Total errors: {int(predictions['is_error'].sum())}",
        "",
        "## Error rate by domain",
        "",
        _markdown_table(domain_rates),
        "",
        "## Error rate by review length",
        "",
        _markdown_table(length_rates),
        "",
        "## Five most confident wrong predictions by confusion pair",
        "",
    ]
    if not pair_tables:
        report_lines.append("_No misclassified predictions._")
    else:
        for pair_name, table in pair_tables.items():
            report_lines.extend([f"### {pair_name}", "", _markdown_table(table), ""])
    report_path = metrics_dir / "error_analysis.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "misclassified_path": misclassified_path,
        "report_path": report_path,
        "domain_rates": domain_rates,
        "length_rates": length_rates,
        "pair_tables": pair_tables,
    }


def main(argv: list[str] | None = None) -> None:
    """Write confusion metrics, error breakdown, and detailed error analysis."""
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
    analysis = error_analysis(cfg)
    print(f"wrote {binary_path}")
    print(f"wrote {errors_path}")
    print(f"wrote {analysis['misclassified_path']}")
    print(f"wrote {analysis['report_path']}")


if __name__ == "__main__":
    main()

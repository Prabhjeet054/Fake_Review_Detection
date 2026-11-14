"""Error analysis on the held-out test split."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import label_names, load_config, parse_config_args, path_from_cfg
from src.embed import load_embeddings
from src.evaluate import compute_metrics, predict_classifier, save_metrics
from src.utils import load_dataframe, save_dataframe, save_json, seed_from_cfg, set_seed, setup_logging


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
    names = label_names(cfg)
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
    source_col = cfg["columns"]["source"]
    if source_col not in df.columns:
        return {}
    out: dict[str, Any] = {}
    for source, group in df.groupby(source_col):
        out[str(source)] = compute_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy(), cfg)
    return out


def main(argv: list[str] | None = None) -> None:
    """Write error tables and source-wise metrics for the test split."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    test_df = load_dataframe(path_from_cfg(cfg, "paths.splits") / "test.csv")
    embeddings, labels, ids = load_embeddings(cfg, "test")
    preds = predict_classifier(embeddings, cfg)
    merged = attach_predictions(test_df, ids, labels, preds, cfg)
    errors = error_slice(merged, cfg)
    metrics_dir = path_from_cfg(cfg, "paths.metrics")
    save_dataframe(errors, metrics_dir / "error_examples.csv")
    by_source = metrics_by_source(merged, cfg)
    for source, metrics in by_source.items():
        save_metrics(metrics, cfg, f"deep_model_test_{source}")
    summary = {
        "n_test": int(len(merged)),
        "n_errors": int(merged["is_error"].sum()),
        "error_rate": float(merged["is_error"].mean()) if len(merged) else 0.0,
        "confusion_pairs": confusion_pairs(merged),
        "errors_by_source": merged.groupby(cfg["columns"]["source"])["is_error"].mean().to_dict()
        if cfg["columns"]["source"] in merged.columns
        else {},
    }
    path = save_json(summary, metrics_dir / "error_analysis.json")
    logger.info("Error analysis: %s errors / %s", summary["n_errors"], summary["n_test"])
    print(f"wrote {path} errors={summary['n_errors']}")


if __name__ == "__main__":
    main()

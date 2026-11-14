"""Exploratory summaries and figures for the processed reviews."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import label_names, load_config, parse_config_args, path_from_cfg
from src.plots import plot_class_distribution, plot_length_distribution
from src.utils import load_dataframe, save_json, seed_from_cfg, set_seed, setup_logging


def length_stats(series: pd.Series) -> dict[str, float]:
    """Compute token-length summary statistics."""
    lengths = series.fillna("").astype(str).str.split().map(len)
    if lengths.empty:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "count": int(lengths.count()),
        "mean": float(lengths.mean()),
        "std": float(lengths.std(ddof=0)),
        "min": float(lengths.min()),
        "median": float(lengths.median()),
        "max": float(lengths.max()),
    }


def summarize_corpus(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-serializable EDA summary."""
    label_col = cfg["columns"]["label"]
    source_col = cfg["columns"]["source"]
    text_col = cfg["columns"]["clean_text"] if cfg["columns"]["clean_text"] in df.columns else cfg["columns"]["text"]
    names = label_names(cfg)
    counts = df[label_col].value_counts().sort_index()
    by_source = pd.crosstab(df[source_col], df[label_col]).to_dict() if source_col in df.columns else {}
    return {
        "n_reviews": int(len(df)),
        "n_groups": int(df[cfg["columns"]["group"]].nunique()) if cfg["columns"]["group"] in df.columns else None,
        "class_counts": {names.get(int(k), str(k)): int(v) for k, v in counts.items()},
        "source_counts": df[source_col].value_counts().to_dict() if source_col in df.columns else {},
        "source_by_label": {str(k): {str(ik): int(iv) for ik, iv in v.items()} for k, v in by_source.items()},
        "length": length_stats(df[text_col]),
        "split_counts": df[cfg["columns"]["split"]].value_counts().to_dict()
        if cfg["columns"]["split"] in df.columns
        else {},
    }


def load_all_splits(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load train/val/test CSVs if present, else the grouped interim file."""
    split_dir = path_from_cfg(cfg, "paths.splits")
    frames: list[pd.DataFrame] = []
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.csv"
        if path.exists():
            frames.append(load_dataframe(path))
    if frames:
        return pd.concat(frames, ignore_index=True)
    grouped = path_from_cfg(cfg, "data.interim_grouped")
    if grouped.exists():
        return load_dataframe(grouped)
    raise FileNotFoundError("No split or grouped data found for EDA.")


def main(argv: list[str] | None = None) -> None:
    """Write EDA metrics JSON and standard figures."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    df = load_all_splits(cfg)
    summary = summarize_corpus(df, cfg)
    metrics_dir = path_from_cfg(cfg, "paths.metrics")
    out_json = save_json(summary, metrics_dir / "eda_summary.json")
    text_col = cfg["columns"]["clean_text"] if cfg["columns"]["clean_text"] in df.columns else cfg["columns"]["text"]
    plot_class_distribution(df, cfg, "eda_class_distribution")
    plot_length_distribution(df, cfg, text_col, "eda_length_distribution")
    logger.info("EDA summary -> %s", out_json)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()

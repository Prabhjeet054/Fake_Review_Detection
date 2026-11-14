"""Tests that eda_summary.json matches pandas recomputation on cleaned.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config
from src.utils import load_json


def _cleaned_and_summary() -> tuple[pd.DataFrame, dict]:
    """Load cleaned.csv and the written EDA JSON."""
    cfg = load_config()
    cleaned_path = Path(cfg.paths.interim_dir) / "cleaned.csv"
    summary_path = Path(cfg.paths.metrics_dir) / "eda_summary.json"
    df = pd.read_csv(cleaned_path)
    summary = load_json(summary_path)
    return df, summary


def test_json_class_counts_match_pandas() -> None:
    """JSON counts_per_class equals pandas value_counts on cleaned.csv."""
    df, summary = _cleaned_and_summary()
    counts = df["label"].value_counts().sort_index()
    for label, count in counts.items():
        json_count = summary["counts_per_class"][str(int(label))]["count"]
        assert json_count == int(count)


def test_json_mean_word_counts_match_pandas() -> None:
    """JSON mean word counts equal pandas means of whitespace-split text_clean."""
    df, summary = _cleaned_and_summary()
    word_count = df["text_clean"].fillna("").astype(str).str.split().map(len)
    means = word_count.groupby(df["label"]).mean()
    for label, mean in means.items():
        json_mean = summary["word_count_per_class"][str(int(label))]["mean"]
        assert json_mean == float(mean)

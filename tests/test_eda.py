"""Tests for src.eda."""

from src.clean import clean_dataframe
from src.dedup import assign_group_ids
from src.eda import length_stats, summarize_corpus


def test_length_stats_and_summary(cfg: dict, reviews_df) -> None:
    """EDA summary includes class counts and length statistics."""
    grouped = assign_group_ids(clean_dataframe(reviews_df, cfg), cfg)
    stats = length_stats(grouped[cfg["columns"]["clean_text"]])
    assert stats["count"] == len(grouped)
    assert stats["mean"] > 0
    summary = summarize_corpus(grouped, cfg)
    assert summary["n_reviews"] == len(grouped)
    assert len(summary["class_counts"]) == 3

"""Tests for src.dedup."""

from src.clean import clean_dataframe
from src.dedup import assign_group_ids, jaccard, shingles


def test_shingles_and_jaccard() -> None:
    """Identical texts have Jaccard 1; disjoint texts have 0."""
    left = shingles("abcdef", 3, "char")
    right = shingles("abcdef", 3, "char")
    assert jaccard(left, right) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0


def test_exact_duplicates_share_group(cfg: dict, reviews_df) -> None:
    """Exact duplicate reviews receive the same group_id."""
    cleaned = clean_dataframe(reviews_df, cfg)
    grouped = assign_group_ids(cleaned, cfg)
    group_col = cfg["columns"]["group"]
    id_col = cfg["columns"]["id"]
    groups = grouped.set_index(id_col)[group_col]
    assert groups.loc["syn_0_00"] == groups.loc["syn_0_dup"]
    assert grouped[group_col].nunique() < len(grouped)

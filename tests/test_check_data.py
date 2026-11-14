"""Tests for src.check_data."""

from src.check_data import check_empty_text, check_labels, checks_failed, run_checks
from src.clean import clean_dataframe
from src.dedup import assign_group_ids
from src.split import assign_splits


def test_clean_splits_pass_checks(cfg: dict, reviews_df) -> None:
    """A properly grouped split has no fatal issues."""
    grouped = assign_group_ids(clean_dataframe(reviews_df, cfg), cfg)
    splits = assign_splits(grouped, cfg)
    issues = run_checks(splits, cfg)
    assert checks_failed(issues, cfg) is False


def test_empty_text_is_flagged(cfg: dict, reviews_df) -> None:
    """Empty cleaned text produces an error issue."""
    frame = clean_dataframe(reviews_df, cfg)
    frame.loc[0, cfg["columns"]["clean_text"]] = "   "
    issues = check_empty_text(frame, cfg)
    assert any(issue["code"] == "empty_text" for issue in issues)


def test_invalid_label_is_flagged(cfg: dict, reviews_df) -> None:
    """Out-of-schema labels are reported."""
    frame = reviews_df.copy()
    frame.loc[0, cfg["columns"]["label"]] = 9
    issues = check_labels(frame, cfg)
    assert any(issue["code"] == "invalid_labels" for issue in issues)

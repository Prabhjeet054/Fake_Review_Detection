"""Tests for src.split."""

from src.clean import clean_dataframe
from src.dedup import assign_group_ids
from src.split import assign_splits, leakage_groups


def test_split_is_group_aware_and_complete(cfg: dict, reviews_df) -> None:
    """Every review is assigned a split and groups do not leak."""
    grouped = assign_group_ids(clean_dataframe(reviews_df, cfg), cfg)
    splits = assign_splits(grouped, cfg)
    assert set(splits) == {"train", "val", "test"}
    total = sum(len(frame) for frame in splits.values())
    assert total == len(grouped)
    overlap = leakage_groups(splits, cfg)
    assert all(len(value) == 0 for value in overlap.values())
    # Duplicate pair stays together.
    id_col = cfg["columns"]["id"]
    split_col = cfg["columns"]["split"]
    locations = {}
    for name, frame in splits.items():
        for review_id in frame[id_col]:
            locations[review_id] = name
    assert locations["syn_0_00"] == locations["syn_0_dup"]
    assert split_col in splits["train"].columns

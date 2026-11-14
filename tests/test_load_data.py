"""Tests for src.load_data."""

from src.load_data import combine_datasets, load_all, load_ott, load_salminen


def test_load_salminen_maps_or_and_cg(cfg: dict) -> None:
    """OR maps to 0 and CG maps to 2."""
    df = load_salminen(cfg)
    labels = set(df[cfg["columns"]["label"]].unique())
    assert labels <= {0, 2}
    assert 0 in labels and 2 in labels
    assert df[cfg["columns"]["source"]].eq("salminen").all()


def test_load_ott_maps_truthful_and_deceptive(cfg: dict) -> None:
    """Ott truthful maps to 0 and deceptive maps to 1."""
    df = load_ott(cfg)
    labels = set(df[cfg["columns"]["label"]].astype(int))
    assert labels == {0, 1}
    assert df[cfg["columns"]["source"]].eq("ott").all()
    assert df[cfg["columns"]["polarity"]].isin(["positive", "negative"]).all()


def test_combine_and_load_all(cfg: dict) -> None:
    """Combined corpus contains all three classes."""
    combined = load_all(cfg)
    assert set(combined[cfg["columns"]["label"]].unique()) == {0, 1, 2}
    again = combine_datasets([load_salminen(cfg), load_ott(cfg)], cfg)
    assert len(again) == len(combined)

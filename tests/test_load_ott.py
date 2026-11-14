"""Tests for load_ott using a 4-row real-format fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.load_data import load_ott

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ott_sample.csv"


def test_truthful_and_deceptive_mapping() -> None:
    """truthful maps to 0 and deceptive maps to 1."""
    df = load_ott(FIXTURE)
    assert len(df) == 4
    assert (df["label"] == 0).sum() == 2
    assert (df["label"] == 1).sum() == 2
    assert set(df["label"].tolist()) == {0, 1}


def test_columns_source_and_domain() -> None:
    """Required columns are present; source is ott and domain is hotel."""
    df = load_ott(FIXTURE)
    for column in ["text", "label", "source", "domain", "category"]:
        assert column in df.columns
    assert (df["source"] == "ott").all()
    assert (df["domain"] == "hotel").all()
    assert set(df["category"]) == {"hilton", "affinia"}

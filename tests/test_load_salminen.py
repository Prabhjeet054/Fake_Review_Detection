"""Tests for load_salminen using a tiny real-format fixture."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.load_data import OUTPUT_COLUMNS, load_salminen

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "salminen_sample.csv"


def test_or_maps_to_zero_cg_maps_to_two() -> None:
    """OR maps to 0 and CG maps to 2."""
    df = load_salminen(FIXTURE)
    raw = pd.read_csv(FIXTURE)
    n_or = int((raw["label"].str.strip().str.upper() == "OR").sum())
    n_cg_with_text = int(
        (
            (raw["label"].str.strip().str.upper() == "CG")
            & raw["text_"].fillna("").astype(str).str.strip().ne("")
        ).sum()
    )
    assert (df["label"] == 0).sum() == n_or
    assert (df["label"] == 2).sum() == n_cg_with_text
    assert set(df["label"].unique()) <= {0, 2}


def test_empty_text_row_is_dropped() -> None:
    """The fixture row with empty text is not in the output."""
    raw = pd.read_csv(FIXTURE)
    n_empty = int(raw["text_"].fillna("").astype(str).str.strip().eq("").sum())
    df = load_salminen(FIXTURE)
    assert n_empty == 1
    assert len(df) == len(raw) - n_empty
    assert df["text"].fillna("").astype(str).str.strip().ne("").all()


def test_output_columns_exact() -> None:
    """Output columns are exactly text, label, source, domain, category."""
    df = load_salminen(FIXTURE)
    assert list(df.columns) == list(OUTPUT_COLUMNS)
    assert list(df.columns) == ["text", "label", "source", "domain", "category"]
    assert (df["source"] == "salminen").all()
    assert (df["domain"] == "product").all()


def test_missing_file_mentions_download() -> None:
    """A missing file raises FileNotFoundError whose message mentions download."""
    missing = Path("/tmp/missing-salminen-file.csv")
    with pytest.raises(FileNotFoundError, match="download"):
        load_salminen(missing)

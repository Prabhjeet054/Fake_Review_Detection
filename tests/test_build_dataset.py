"""Tests for build_dataset using tiny real-format fixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.load_data import build_dataset
from src.schema import REQUIRED_COLUMNS, validate_dataframe

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _md5(path: Path) -> str:
    """Return the MD5 hex digest of a file."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _fixture_config(tmp_path: Path, genuine: int = 2, ai: int = 2):
    """Return a config that loads fixtures and writes merged.csv under tmp_path."""
    cfg = load_config()
    cfg.data.salminen_file = FIXTURES / "salminen_sample.csv"
    cfg.data.ott_file = FIXTURES / "ott_sample.csv"
    cfg.data.genuine_product_sample = genuine
    cfg.data.ai_sample = ai
    cfg.paths.interim_dir = tmp_path
    return cfg


def test_keeps_all_ott_and_samples_salminen(tmp_path: Path) -> None:
    """All Ott rows are kept; Salminen OR/CG counts match the patched sample sizes."""
    df = build_dataset(_fixture_config(tmp_path, genuine=2, ai=2))
    counts = (
        pd.crosstab(df["label"], df["domain"])
        .reindex(index=[0, 1, 2], columns=["hotel", "product"], fill_value=0)
    )
    assert len(df) == 8
    assert int(counts.loc[0, "hotel"]) == 2
    assert int(counts.loc[0, "product"]) == 2
    assert int(counts.loc[1, "hotel"]) == 2
    assert int(counts.loc[1, "product"]) == 0
    assert int(counts.loc[2, "hotel"]) == 0
    assert int(counts.loc[2, "product"]) == 2
    assert (df["source"] == "ott").sum() == 4
    assert ((df["source"] == "salminen") & (df["label"] == 0)).sum() == 2
    assert ((df["source"] == "salminen") & (df["label"] == 2)).sum() == 2


def test_ids_unique_no_nan_and_schema_valid(tmp_path: Path) -> None:
    """Ids are unique, there are no NaNs, and schema validation passes."""
    df = build_dataset(_fixture_config(tmp_path))
    for column in REQUIRED_COLUMNS:
        assert column in df.columns
    assert df["id"].is_unique
    assert df["id"].nunique() == len(df)
    assert df["id"].dtype.kind == "i"
    assert not df.isna().any().any()
    validate_dataframe(df)
    saved = tmp_path / "merged.csv"
    assert saved.is_file()
    reloaded = pd.read_csv(saved)
    assert not reloaded.isna().any().any()
    validate_dataframe(reloaded)


def test_two_runs_write_identical_merged_csv(tmp_path: Path) -> None:
    """Two seeded builds write merged.csv files with the same MD5 hash."""
    first_dir = tmp_path / "run1"
    second_dir = tmp_path / "run2"
    first_dir.mkdir()
    second_dir.mkdir()
    build_dataset(_fixture_config(first_dir))
    build_dataset(_fixture_config(second_dir))
    assert _md5(first_dir / "merged.csv") == _md5(second_dir / "merged.csv")

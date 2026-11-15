"""Tests for src.load_data.load_salminen."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.load_data import load_salminen

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "salminen_sample.csv"


def test_load_salminen_maps_or_and_cg() -> None:
    """OR maps to 0 and CG maps to 2 with source/domain set."""
    df = load_salminen(FIXTURE)
    assert set(df.columns) == {"text", "label", "source", "domain", "category"}
    assert set(df["label"].unique()) == {0, 2}
    assert (df["source"] == "salminen").all()
    assert (df["domain"] == "product").all()
    assert df["text"].str.strip().ne("").all()


def test_load_salminen_prints_columns(capsys: pytest.CaptureFixture[str]) -> None:
    """The loader prints the real CSV header and sample rows."""
    load_salminen(FIXTURE)
    captured = capsys.readouterr().out
    assert "text_" in captured
    assert "Salminen columns:" in captured
    assert "Salminen sample rows:" in captured


def test_load_salminen_drops_other_labels_and_empty_text(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Unknown labels and empty text are dropped and counted in the log."""
    path = tmp_path / "salminen.csv"
    path.write_text(
        "category,rating,label,text_\n"
        "Kitchen,5.0,OR,Good blender\n"
        "Kitchen,1.0,XX,Should drop this label\n"
        "Home,5.0,CG,   \n"
        "Home,4.0,CG,Nice lamp\n",
        encoding="utf-8",
    )
    caplog.set_level("INFO")
    df = load_salminen(path)
    assert len(df) == 2
    assert set(df["label"]) == {0, 2}


def test_missing_file_raises() -> None:
    """A missing CSV raises FileNotFoundError with the configured filename."""
    missing = Path("/tmp/definitely-missing-salminen.csv")
    with pytest.raises(FileNotFoundError, match="fake_reviews_dataset.csv"):
        load_salminen(missing)

"""Tests for src.clean using the specified string cases and a short-row drop."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.clean import clean_dataframe, clean_text
from src.config import load_config

EXACT_CASES = [
    ("<br>Great fit", "great fit"),
    ("I don't like it", "i do not like it"),
    ("good   value", "good value"),
    ("AMAZING product!!", "amazing product!!"),
    ("Love it ★★★", "love it"),
    ("Fish &amp; chips", "fish chips"),
    ("", ""),
]


@pytest.mark.parametrize("raw, expected", EXACT_CASES)
def test_clean_text_exact_cases(raw: str, expected: str) -> None:
    """Each specified input maps to the expected cleaned string."""
    assert clean_text(raw) == expected


def test_clean_text_is_idempotent_on_real_reviews() -> None:
    """Cleaning 50 random real reviews twice equals cleaning them once."""
    cfg = load_config()
    merged_path = Path(cfg.paths.interim_dir) / "merged.csv"
    df = pd.read_csv(merged_path)
    sample = df["text"].dropna().sample(n=50, random_state=int(cfg.seed))
    for text in sample:
        once = clean_text(str(text))
        twice = clean_text(once)
        assert twice == once


def test_clean_dataframe_drops_three_word_review_when_min_words_is_five(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 3-word review is dropped when min_words=5."""
    cfg = load_config()
    cfg.paths.interim_dir = tmp_path
    monkeypatch.setattr("src.clean.load_config", lambda: cfg)
    df = pd.DataFrame(
        {
            "id": [0, 1],
            "text": ["one two three", "this review has more than five words"],
            "label": [0, 1],
            "source": ["ott", "salminen"],
            "domain": ["hotel", "product"],
        }
    )
    out = clean_dataframe(df, min_words=5)
    assert len(out) == 1
    assert out.iloc[0]["id"] == 1
    assert "one two three" not in set(out["text"])
    assert out["text_clean"].str.split().map(len).min() >= 5

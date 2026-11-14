"""Tests for src.clean."""

from src.clean import clean_dataframe, clean_text, collapse_whitespace, expand_contractions, strip_html


def test_strip_html_and_whitespace() -> None:
    """HTML is removed and whitespace is squeezed."""
    assert "Hello" in strip_html("<p>Hello</p>")
    assert collapse_whitespace(" a   b\n c ") == "a b c"


def test_expand_contractions() -> None:
    """Common contractions expand using the mapping."""
    assert "do not" in expand_contractions("don't go", {"don't": "do not"}).lower()


def test_clean_text_pipeline(cfg: dict) -> None:
    """Cleaning lowercases, drops symbols, and keeps allowed punctuation."""
    text = "<b>Don't</b> buy this!!! It's #1, really?"
    cleaned = clean_text(text, cfg)
    assert "<b>" not in cleaned
    assert "#" not in cleaned
    assert "do not" in cleaned
    assert cleaned == cleaned.lower()
    assert "!" in cleaned or "?" in cleaned or "," in cleaned


def test_clean_dataframe_adds_column(cfg: dict, reviews_df) -> None:
    """DataFrame cleaning writes the configured clean-text column."""
    out = clean_dataframe(reviews_df, cfg)
    clean_col = cfg["columns"]["clean_text"]
    assert clean_col in out.columns
    assert out[clean_col].str.len().min() > 0

"""Text cleaning: HTML, symbols, contractions, casing, and whitespace."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from src.config import load_config, parse_config_args, path_from_cfg
from src.utils import load_dataframe, save_dataframe, seed_from_cfg, set_seed, setup_logging

_WHITESPACE_RE = re.compile(r"\s+")


@lru_cache(maxsize=4)
def load_contractions(path: str) -> dict[str, str]:
    """Load a contractions mapping, keyed in lowercase."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(key).lower(): str(value) for key, value in raw.items()}


def strip_html(text: str) -> str:
    """Remove HTML tags and return visible text."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


def normalize_apostrophes(text: str, apostrophe_chars: list[str], replacement: str = "'") -> str:
    """Rewrite unicode apostrophes to a single ASCII apostrophe."""
    for char in apostrophe_chars:
        if char != replacement:
            text = text.replace(char, replacement)
    return text


def expand_contractions(text: str, mapping: dict[str, str]) -> str:
    """Replace contractions using the provided mapping."""
    if not mapping:
        return text
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(key) for key in sorted(mapping, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )

    def _replace(match: re.Match[str]) -> str:
        original = match.group(0)
        replacement = mapping[original.lower()]
        if original[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    return pattern.sub(_replace, text)


def remove_special_symbols(text: str, keep_chars: str, extra_to_space: str) -> str:
    """Drop symbols except alphanumerics, whitespace, and configured keep-chars."""
    for char in extra_to_space:
        text = text.replace(char, " ")
    keep = set(keep_chars)
    cleaned: list[str] = []
    for char in text:
        if char.isalnum() or char.isspace() or char in keep:
            cleaned.append(char)
    return "".join(cleaned)


def collapse_whitespace(text: str) -> str:
    """Squeeze repeated whitespace and trim ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str, cfg: dict[str, Any], contractions: dict[str, str] | None = None) -> str:
    """Apply the configured cleaning pipeline to a single string."""
    settings = cfg["preprocessing"]
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = normalize_apostrophes(cleaned, list(settings["apostrophe_chars"]))
    if settings["strip_html"]:
        cleaned = strip_html(cleaned)
    cleaned = remove_special_symbols(cleaned, str(settings["keep_chars"]), str(settings["extra_to_space"]))
    if settings["expand_contractions"]:
        mapping = contractions if contractions is not None else load_contractions(
            str(path_from_cfg(cfg, "paths.contractions"))
        )
        cleaned = expand_contractions(cleaned, mapping)
    if settings["lowercase"]:
        cleaned = cleaned.lower()
    if settings["collapse_whitespace"]:
        cleaned = collapse_whitespace(cleaned)
    return cleaned


def clean_dataframe(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add a cleaned-text column and optionally drop empty reviews."""
    mapping = load_contractions(str(path_from_cfg(cfg, "paths.contractions")))
    text_col = cfg["columns"]["text"]
    clean_col = cfg["columns"]["clean_text"]
    out = df.copy()
    out[clean_col] = out[text_col].map(lambda value: clean_text(value, cfg, mapping))
    if cfg["preprocessing"]["drop_empty"]:
        token_counts = out[clean_col].str.split().map(len)
        out = out.loc[token_counts >= int(cfg["preprocessing"]["min_tokens"])].reset_index(drop=True)
    return out


def main(argv: list[str] | None = None) -> None:
    """Clean the interim raw reviews file."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    raw_path = path_from_cfg(cfg, "data.interim_raw")
    df = load_dataframe(raw_path)
    cleaned = clean_dataframe(df, cfg)
    out_path = path_from_cfg(cfg, "data.interim_clean")
    save_dataframe(cleaned, out_path)
    logger.info("Cleaned %s reviews -> %s", len(cleaned), out_path)
    print(f"wrote {out_path} rows={len(cleaned)}")


if __name__ == "__main__":
    main()

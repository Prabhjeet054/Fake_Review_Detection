"""Clean review text and drop rows that are too short after cleaning."""

from __future__ import annotations

import html
import re
from pathlib import Path

import contractions
import pandas as pd
from bs4 import BeautifulSoup

from src.config import load_config
from src.utils import ensure_dirs, get_logger, set_seed

KEEP_CHARS = set(".,!?'")
WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Remove HTML tags and unescape HTML entities."""
    visible = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return html.unescape(visible)


def _keep_allowed(text: str) -> str:
    """Keep letters, digits, whitespace, and . , ! ? ' ; drop other symbols."""
    return "".join(ch for ch in text if ch.isalnum() or ch.isspace() or ch in KEEP_CHARS)


def clean_text(text: str) -> str:
    """Clean one review without stemming or stop-word removal."""
    if text is None:
        return ""
    cleaned = _strip_html(str(text))
    cleaned = contractions.fix(cleaned)
    cleaned = _keep_allowed(cleaned)
    cleaned = cleaned.lower()
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def _word_count(text: str) -> int:
    """Return the number of whitespace-separated tokens."""
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def clean_dataframe(df: pd.DataFrame, min_words: int) -> pd.DataFrame:
    """Add text_clean, drop short rows, log drops per label, and save cleaned.csv."""
    logger = get_logger(__name__)
    cfg = load_config()
    out = df.copy()
    out["text_clean"] = out["text"].map(lambda value: clean_text("" if pd.isna(value) else str(value)))
    n_words = out["text_clean"].map(_word_count)
    short = n_words < int(min_words)
    n_dropped = int(short.sum())
    if n_dropped and "label" in out.columns:
        dropped = out.loc[short, "label"].value_counts().sort_index()
        for label, count in dropped.items():
            logger.info(
                "Dropped %s rows with fewer than %s words after cleaning (label=%s)",
                int(count),
                min_words,
                int(label),
            )
    else:
        logger.info("Dropped %s rows with fewer than %s words after cleaning", n_dropped, min_words)
    out = out.loc[~short].reset_index(drop=True)
    ensure_dirs(cfg)
    out_path = Path(cfg.paths.interim_dir) / "cleaned.csv"
    out.to_csv(out_path, index=False)
    print(f"saved={out_path} n_rows={len(out)}")
    return out


def main() -> None:
    """Clean data/interim/merged.csv using config.data.min_words."""
    cfg = load_config()
    set_seed(int(cfg.seed))
    merged_path = Path(cfg.paths.interim_dir) / "merged.csv"
    df = pd.read_csv(merged_path)
    clean_dataframe(df, int(cfg.data.min_words))


if __name__ == "__main__":
    main()

"""Load Salminen and Ott review datasets into the canonical schema."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import AttrDict, load_config
from src.schema import validate_dataframe
from src.utils import ensure_dirs, get_logger, set_seed

SALMINEN_LABEL_MAP = {"OR": 0, "CG": 2}
OTT_LABEL_MAP = {"TRUTHFUL": 0, "DECEPTIVE": 1}
OUTPUT_COLUMNS = ["text", "label", "source", "domain", "category"]


def _pick_column(columns: list[str], *candidates: str) -> str:
    """Return the first candidate that exists in the CSV header (case-insensitive)."""
    lookup = {name.strip().lower(): name for name in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    raise ValueError(
        f"None of {list(candidates)} found in CSV columns {columns}. "
        "Inspect the printed header and update the loader to match the file."
    )


def _require_csv(path: str | Path, dataset_name: str, expected: Path, download_hint: str) -> Path:
    """Return path if it exists; otherwise raise FileNotFoundError with download instructions."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{dataset_name} dataset not found at {csv_path}. "
            f"Please download {download_hint} "
            f"and save it using the expected filename from config: {expected}"
        )
    return csv_path


def _read_preview(csv_path: Path, title: str) -> pd.DataFrame:
    """Read a CSV, strip headers, and print the real columns plus three sample rows."""
    raw = pd.read_csv(csv_path)
    raw.columns = [str(col).strip() for col in raw.columns]
    print(f"{title} columns:", list(raw.columns))
    print(f"{title} sample rows:")
    print(raw.head(3).to_string(index=False))
    return raw


def _drop_empty_text(df: pd.DataFrame, logger: logging.Logger, dataset_name: str) -> pd.DataFrame:
    """Drop rows whose text is NaN or blank after strip, and log the count."""
    empty_text = df["text"].isna() | df["text"].astype(str).str.strip().eq("")
    n_dropped_text = int(empty_text.sum())
    if n_dropped_text:
        logger.info("Dropped %s %s rows with missing or empty text", n_dropped_text, dataset_name)
    return df.loc[~empty_text].copy()


def load_salminen(path: str | Path) -> pd.DataFrame:
    """Load Salminen reviews, mapping OR->0 and CG->2 into product-domain rows."""
    logger = get_logger(__name__)
    cfg = load_config()
    csv_path = _require_csv(
        path,
        "Salminen",
        cfg.data.salminen_file,
        "the Fake Reviews Dataset from https://osf.io/tyue9/",
    )
    raw = _read_preview(csv_path, "Salminen")
    text_col = _pick_column(list(raw.columns), "text_", "text")
    label_col = _pick_column(list(raw.columns), "label")
    category_col = _pick_column(list(raw.columns), "category")
    labels = raw[label_col].astype(str).str.strip().str.upper()
    known = labels.isin(SALMINEN_LABEL_MAP)
    n_dropped_labels = int((~known).sum())
    if n_dropped_labels:
        logger.info("Dropped %s Salminen rows with labels outside OR/CG", n_dropped_labels)
    kept = raw.loc[known].copy()
    kept["text"] = kept[text_col]
    kept["label"] = labels.loc[known].map(SALMINEN_LABEL_MAP).astype(int)
    kept = _drop_empty_text(kept, logger, "Salminen")
    kept["source"] = "salminen"
    kept["domain"] = "product"
    kept["category"] = kept[category_col]
    return kept[OUTPUT_COLUMNS].reset_index(drop=True)


def load_ott(path: str | Path) -> pd.DataFrame:
    """Load Ott reviews, mapping truthful->0 and deceptive->1 into hotel-domain rows."""
    logger = get_logger(__name__)
    cfg = load_config()
    csv_path = _require_csv(
        path,
        "Ott",
        cfg.data.ott_file,
        "the Deceptive Opinion Spam Corpus (deceptive-opinion.csv) from "
        "https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus",
    )
    raw = _read_preview(csv_path, "Ott")
    text_col = _pick_column(list(raw.columns), "text", "review", "text_")
    label_col = _pick_column(list(raw.columns), "deceptive", "label", "class")
    hotel_col = _pick_column(list(raw.columns), "hotel", "category")
    polarity_col = None
    try:
        polarity_col = _pick_column(list(raw.columns), "polarity")
    except ValueError:
        polarity_col = None
    labels = raw[label_col].astype(str).str.strip().str.upper()
    known = labels.isin(OTT_LABEL_MAP)
    n_dropped_labels = int((~known).sum())
    if n_dropped_labels:
        logger.info("Dropped %s Ott rows with labels outside truthful/deceptive", n_dropped_labels)
    kept = raw.loc[known].copy()
    kept["text"] = kept[text_col]
    kept["label"] = labels.loc[known].map(OTT_LABEL_MAP).astype(int)
    kept = _drop_empty_text(kept, logger, "Ott")
    kept["source"] = "ott"
    kept["domain"] = "hotel"
    kept["category"] = kept[hotel_col]
    out = kept[OUTPUT_COLUMNS].copy()
    if polarity_col is not None:
        out["polarity"] = kept[polarity_col].to_numpy()
    return out.reset_index(drop=True)


def _sample_rows(df: pd.DataFrame, n: int, seed: int, name: str) -> pd.DataFrame:
    """Return n rows sampled without replacement using the given seed."""
    available = len(df)
    if n > available:
        raise ValueError(f"Requested {n} {name} rows but only {available} are available.")
    return df.sample(n=n, random_state=seed)


def build_dataset(config: AttrDict) -> pd.DataFrame:
    """Merge all Ott rows with a seeded Salminen sample, validate, and save merged.csv."""
    seed = int(config.seed)
    set_seed(seed)
    ott = load_ott(config.data.ott_file)[OUTPUT_COLUMNS]
    salminen = load_salminen(config.data.salminen_file)[OUTPUT_COLUMNS]
    genuine = _sample_rows(
        salminen.loc[salminen["label"] == 0],
        int(config.data.genuine_product_sample),
        seed,
        "Salminen genuine (OR)",
    )
    ai = _sample_rows(
        salminen.loc[salminen["label"] == 2],
        int(config.data.ai_sample),
        seed,
        "Salminen AI (CG)",
    )
    merged = pd.concat([ott, genuine, ai], ignore_index=True)
    merged = merged.sample(frac=1, random_state=seed).reset_index(drop=True)
    merged.insert(0, "id", range(len(merged)))
    merged["id"] = merged["id"].astype(int)
    validate_dataframe(merged)
    ensure_dirs(config)
    out_path = Path(config.paths.interim_dir) / "merged.csv"
    merged.to_csv(out_path, index=False)
    print("crosstab label x domain:")
    print(pd.crosstab(merged["label"], merged["domain"]))
    print(f"saved={out_path} n_rows={len(merged)}")
    return merged


def main() -> None:
    """Build the merged dataset from config.yaml."""
    build_dataset(load_config())


if __name__ == "__main__":
    main()

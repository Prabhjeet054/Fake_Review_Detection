"""Load the Salminen fake-reviews dataset into the canonical schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config
from src.utils import get_logger

SALMINEN_LABEL_MAP = {"OR": 0, "CG": 2}
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


def load_salminen(path: str | Path) -> pd.DataFrame:
    """Load Salminen reviews, mapping OR->0 and CG->2 into product-domain rows."""
    logger = get_logger(__name__)
    csv_path = Path(path)
    if not csv_path.is_file():
        cfg = load_config()
        expected = cfg.data.salminen_file
        raise FileNotFoundError(
            f"Salminen dataset not found at {csv_path}. "
            "Please download the Fake Reviews Dataset from https://osf.io/tyue9/ "
            f"and save it using the expected filename from config: {expected}"
        )
    raw = pd.read_csv(csv_path)
    raw.columns = [str(col).strip() for col in raw.columns]
    print("Salminen columns:", list(raw.columns))
    print("Salminen sample rows:")
    print(raw.head(3).to_string(index=False))
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
    empty_text = kept["text"].isna() | kept["text"].astype(str).str.strip().eq("")
    n_dropped_text = int(empty_text.sum())
    if n_dropped_text:
        logger.info("Dropped %s Salminen rows with missing or empty text", n_dropped_text)
    kept = kept.loc[~empty_text].copy()
    kept["source"] = "salminen"
    kept["domain"] = "product"
    kept["category"] = kept[category_col]
    return kept[OUTPUT_COLUMNS].reset_index(drop=True)


def main() -> None:
    """Load the configured Salminen file and print label counts plus sample reviews."""
    from src.schema import ID2LABEL
    from src.utils import set_seed

    cfg = load_config()
    set_seed(int(cfg.seed))
    df = load_salminen(cfg.data.salminen_file)
    print("rows_per_label:")
    print(df["label"].value_counts().sort_index().to_string())
    print(f"n_rows={len(df)}")
    for label_id, group in df.groupby("label"):
        name = ID2LABEL.get(int(label_id), str(label_id))
        print(f"\n--- label {label_id} ({name}), n={len(group)} ---")
        sample = group.sample(n=min(2, len(group)), random_state=int(cfg.seed))
        for _, row in sample.iterrows():
            print(f"[{row['category']}] {row['text'][:400]}")


if __name__ == "__main__":
    main()

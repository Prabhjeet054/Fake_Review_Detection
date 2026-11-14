"""Canonical label maps and DataFrame validation for reviews."""

from __future__ import annotations

import pandas as pd

from src.config import load_config

_cfg = load_config()
ID2LABEL: dict[int, str] = {int(key): str(value) for key, value in _cfg.labels.items()}
LABEL2ID: dict[str, int] = {name: label_id for label_id, name in ID2LABEL.items()}
REQUIRED_COLUMNS: list[str] = ["id", "text", "label", "source", "domain"]
ALLOWED_LABELS: set[int] = set(ID2LABEL)


def validate_dataframe(df: pd.DataFrame, required: list[str] = REQUIRED_COLUMNS) -> pd.DataFrame:
    """Raise ValueError if the review frame is invalid; otherwise return it unchanged."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    if "label" in df.columns:
        labels = df["label"]
        if labels.isna().any():
            raise ValueError("Column 'label' contains missing values.")
        invalid = sorted({int(value) for value in labels} - ALLOWED_LABELS)
        if invalid:
            raise ValueError(
                f"Column 'label' contains values outside {sorted(ALLOWED_LABELS)}: {invalid}"
            )
    if "text" in df.columns:
        empty_mask = df["text"].isna() | df["text"].astype(str).str.strip().eq("")
        if empty_mask.any():
            raise ValueError("Column 'text' contains NaN or empty values after strip.")
    if "id" in df.columns and df["id"].duplicated().any():
        raise ValueError("Column 'id' values are not unique.")
    return df


def main(argv: list[str] | None = None) -> None:
    """Print the label maps and required columns."""
    print("LABEL2ID=", LABEL2ID)
    print("ID2LABEL=", ID2LABEL)
    print("REQUIRED_COLUMNS=", REQUIRED_COLUMNS)


if __name__ == "__main__":
    main()

"""Canonical review schema and DataFrame validation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import label_names, load_config, parse_config_args


def required_columns(cfg: dict[str, Any]) -> list[str]:
    """Return required column names from config."""
    return [str(col) for col in cfg["schema"]["required_columns"]]


def allowed_labels(cfg: dict[str, Any]) -> set[int]:
    """Return the set of valid integer labels."""
    return {int(value) for value in cfg["schema"]["allowed_labels"]}


def add_label_names(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Attach human-readable label names from config."""
    label_col = cfg["columns"]["label"]
    name_col = cfg["columns"]["label_name"]
    mapping = label_names(cfg)
    out = df.copy()
    out[name_col] = out[label_col].map(mapping)
    return out


def validate_reviews(df: pd.DataFrame, cfg: dict[str, Any], *, require_all: bool = True) -> pd.DataFrame:
    """Validate required columns, labels, and basic types; return a copy."""
    if df.empty and require_all:
        raise ValueError("Review DataFrame is empty.")
    missing = [col for col in required_columns(cfg) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df.copy()
    label_col = cfg["columns"]["label"]
    out[label_col] = out[label_col].astype(int)
    valid = allowed_labels(cfg)
    invalid = sorted(set(out[label_col].unique()) - valid)
    if invalid:
        raise ValueError(f"Invalid labels {invalid}; expected {sorted(valid)}")
    id_col = cfg["columns"]["id"]
    out[id_col] = out[id_col].astype(str)
    text_col = cfg["columns"]["text"]
    out[text_col] = out[text_col].fillna("").astype(str)
    if cfg["columns"]["label_name"] not in out.columns:
        out = add_label_names(out, cfg)
    return out


def empty_reviews_frame(cfg: dict[str, Any]) -> pd.DataFrame:
    """Return an empty DataFrame with the canonical columns."""
    cols = required_columns(cfg) + [
        cfg["columns"]["label_name"],
        cfg["columns"]["category"],
        cfg["columns"]["rating"],
        cfg["columns"]["polarity"],
        cfg["columns"]["hotel"],
    ]
    unique_cols = list(dict.fromkeys(cols))
    return pd.DataFrame(columns=unique_cols)


def main(argv: list[str] | None = None) -> None:
    """Print the canonical schema described by the config."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    print("required_columns:", required_columns(cfg))
    print("allowed_labels:", sorted(allowed_labels(cfg)))
    print("label_names:", label_names(cfg))


if __name__ == "__main__":
    main()

"""Stratified, group-aware train/val/test split."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, parse_config_args, path_from_cfg
from src.utils import ensure_dir, load_dataframe, save_dataframe, seed_from_cfg, set_seed, setup_logging


def _group_labels(df: pd.DataFrame, group_col: str, label_col: str) -> pd.Series:
    """One label per group, using the modal label inside the group."""
    return df.groupby(group_col)[label_col].agg(lambda values: values.mode().iloc[0])


def split_groups(
    group_ids: list[str],
    labels: list[int],
    cfg: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """Split group ids 70/15/15 with label stratification."""
    seed = seed_from_cfg(cfg)
    train_ratio = float(cfg["split"]["train"])
    val_ratio = float(cfg["split"]["val"])
    test_ratio = float(cfg["split"]["test"])
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.")
    holdout = 1.0 - train_ratio
    stratify: list[int] | None = labels
    try:
        train_ids, temp_ids, _, temp_labels = train_test_split(
            group_ids,
            labels,
            test_size=holdout,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_ids, temp_ids, _, temp_labels = train_test_split(
            group_ids,
            labels,
            test_size=holdout,
            random_state=seed,
            stratify=None,
        )
    relative_test = test_ratio / (val_ratio + test_ratio)
    try:
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=relative_test,
            random_state=seed,
            stratify=temp_labels,
        )
    except ValueError:
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=relative_test,
            random_state=seed,
            stratify=None,
        )
    return list(train_ids), list(val_ids), list(test_ids)


def assign_splits(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Return train/val/test frames with a split column, keeping groups intact."""
    group_col = cfg["columns"]["group"]
    label_col = cfg["columns"]["label"]
    split_col = cfg["columns"]["split"]
    if group_col not in df.columns:
        raise ValueError("group_id is required for a group-aware split.")
    labels_by_group = _group_labels(df, group_col, label_col)
    train_ids, val_ids, test_ids = split_groups(
        labels_by_group.index.astype(str).tolist(),
        labels_by_group.astype(int).tolist(),
        cfg,
    )
    lookup = {group: "train" for group in train_ids}
    lookup.update({group: "val" for group in val_ids})
    lookup.update({group: "test" for group in test_ids})
    out = df.copy()
    out[split_col] = out[group_col].astype(str).map(lookup)
    if out[split_col].isna().any():
        raise RuntimeError("Some reviews were not assigned a split.")
    return {
        "train": out.loc[out[split_col] == "train"].reset_index(drop=True),
        "val": out.loc[out[split_col] == "val"].reset_index(drop=True),
        "test": out.loc[out[split_col] == "test"].reset_index(drop=True),
    }


def leakage_groups(splits: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, set[str]]:
    """Return overlapping group ids between split pairs."""
    group_col = cfg["columns"]["group"]
    groups = {name: set(frame[group_col].astype(str)) for name, frame in splits.items()}
    return {
        "train_val": groups["train"] & groups["val"],
        "train_test": groups["train"] & groups["test"],
        "val_test": groups["val"] & groups["test"],
    }


def main(argv: list[str] | None = None) -> None:
    """Write stratified group-aware split CSVs."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    df = load_dataframe(path_from_cfg(cfg, "data.interim_grouped"))
    splits = assign_splits(df, cfg)
    overlap = leakage_groups(splits, cfg)
    if any(overlap.values()):
        raise RuntimeError(f"Group leakage detected: { {k: len(v) for k, v in overlap.items()} }")
    out_dir = ensure_dir(path_from_cfg(cfg, "paths.splits"))
    for name, frame in splits.items():
        path = save_dataframe(frame, out_dir / f"{name}.csv")
        logger.info("%s split: %s rows -> %s", name, len(frame), path)
        print(f"{name}: {len(frame)} -> {path}")


if __name__ == "__main__":
    main()

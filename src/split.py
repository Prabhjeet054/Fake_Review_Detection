# The test split must not be read by any training code.
"""Stratified, group-aware 70/15/15 split via StratifiedGroupKFold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import AttrDict, load_config
from src.utils import ensure_dirs, save_json, set_seed

N_SPLITS = 20
TEST_FOLDS = {0, 1, 2}
VAL_FOLDS = {3, 4, 5}


def _fold_to_split(fold: int) -> str:
    """Map a fold index onto train, val, or test."""
    if fold in TEST_FOLDS:
        return "test"
    if fold in VAL_FOLDS:
        return "val"
    return "train"


def assign_folds(df: pd.DataFrame, seed: int) -> np.ndarray:
    """Return a fold id in 0..19 for every row using StratifiedGroupKFold."""
    y = df["label"].to_numpy()
    groups = df["group_id"].to_numpy()
    fold = np.full(len(df), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=int(seed))
    for fold_id, (_, holdout_idx) in enumerate(splitter.split(df, y=y, groups=groups)):
        fold[holdout_idx] = int(fold_id)
    if (fold < 0).any():
        raise RuntimeError("Some rows were not assigned a StratifiedGroupKFold fold.")
    return fold


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Return label -> row count as JSON-friendly ints."""
    counts = frame["label"].value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def split_summary(splits: dict[str, pd.DataFrame]) -> dict:
    """Build row counts and class counts per split."""
    return {
        name: {"n_rows": int(len(frame)), "class_counts": _class_counts(frame)}
        for name, frame in splits.items()
    }


def _assert_no_group_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if the same group_id appears in more than one split."""
    groups = {name: set(frame["group_id"].tolist()) for name, frame in splits.items()}
    overlap = {
        "train_val": groups["train"] & groups["val"],
        "train_test": groups["train"] & groups["test"],
        "val_test": groups["val"] & groups["test"],
    }
    leaked = {name: len(ids) for name, ids in overlap.items() if ids}
    if leaked:
        raise RuntimeError(f"group_id leakage across splits: {leaked}")


def split_dataset(df: pd.DataFrame, config: AttrDict | None = None) -> dict[str, pd.DataFrame]:
    """Assign 3 folds to test, 3 to val, 14 to train, then save CSVs and JSON."""
    cfg = config if config is not None else load_config()
    seed = int(cfg.seed)
    set_seed(seed)
    work = df.copy()
    work["split"] = [_fold_to_split(int(fold)) for fold in assign_folds(work, seed)]
    splits = {
        "train": work.loc[work["split"] == "train"].reset_index(drop=True),
        "val": work.loc[work["split"] == "val"].reset_index(drop=True),
        "test": work.loc[work["split"] == "test"].reset_index(drop=True),
    }
    _assert_no_group_leakage(splits)
    ensure_dirs(cfg)
    split_dir = Path(cfg.paths.splits_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(split_dir / f"{name}.csv", index=False)
        print(f"{name}: n_rows={len(frame)}")
    summary = split_summary(splits)
    save_json(summary, Path(cfg.paths.metrics_dir) / "split_summary.json")
    print(f"wrote {Path(cfg.paths.metrics_dir) / 'split_summary.json'}")
    return splits


def main() -> None:
    """Split data/processed/dataset.csv into train/val/test CSVs."""
    cfg = load_config()
    path = Path(cfg.paths.processed_dir) / "dataset.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run python -m src.dedup first."
        )
    split_dataset(pd.read_csv(path), cfg)


if __name__ == "__main__":
    main()

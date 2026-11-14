"""Tests for TF-IDF features, including a train-only vocabulary check."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config
from src.features import make_tfidf_features


def _zebra_fixture(split_dir: Path) -> None:
    """Write tiny splits where 'zebra' appears only in the test file."""
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": [0, 1, 2],
            "label": [0, 1, 2],
            "text_clean": [
                "the hotel room was clean and the staff were helpful",
                "the hotel stay was terrible and the room smelled bad",
                "this blender is great quality and i highly recommend it",
            ],
        }
    ).to_csv(split_dir / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [3],
            "label": [0],
            "text_clean": ["the hotel room was quiet and the staff were kind"],
        }
    ).to_csv(split_dir / "val.csv", index=False)
    pd.DataFrame(
        {
            "id": [4],
            "label": [2],
            "text_clean": ["this blender is great and a zebra appears only here"],
        }
    ).to_csv(split_dir / "test.csv", index=False)


def _tiny_config(tmp_path: Path):
    """Return a config that reads the zebra fixture and writes tfidf.joblib under tmp_path."""
    cfg = load_config()
    cfg.tfidf.min_df = 1
    cfg.tfidf.max_features = 500
    cfg.paths.splits_dir = tmp_path / "splits"
    cfg.paths.models_dir = tmp_path / "models"
    _zebra_fixture(cfg.paths.splits_dir)
    return cfg


def test_zebra_only_in_test_is_absent_from_vocabulary(tmp_path: Path) -> None:
    """A word that appears only in the test split is not in the fitted vocabulary."""
    cfg = _tiny_config(tmp_path)
    make_tfidf_features(cfg)
    vectorizer = joblib.load(tmp_path / "models" / "tfidf.joblib")
    vocab = set(vectorizer.get_feature_names_out())
    assert "zebra" not in vocab
    assert "hotel" in vocab


def test_matrix_columns_match_and_rows_match_split_files(tmp_path: Path) -> None:
    """X_train, X_val, and X_test share column count; rows match the split CSVs."""
    cfg = _tiny_config(tmp_path)
    X_train, X_val, X_test, y_train, y_val, y_test = make_tfidf_features(cfg)
    train_n = len(pd.read_csv(cfg.paths.splits_dir / "train.csv"))
    val_n = len(pd.read_csv(cfg.paths.splits_dir / "val.csv"))
    test_n = len(pd.read_csv(cfg.paths.splits_dir / "test.csv"))
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    assert X_train.shape[1] > 0
    assert X_train.shape[0] == train_n == len(y_train)
    assert X_val.shape[0] == val_n == len(y_val)
    assert X_test.shape[0] == test_n == len(y_test)


def test_saved_vectorizer_transforms_new_sentence(tmp_path: Path) -> None:
    """tfidf.joblib loads and maps a new sentence to the same number of columns."""
    cfg = _tiny_config(tmp_path)
    X_train, _, _, _, _, _ = make_tfidf_features(cfg)
    vectorizer = joblib.load(tmp_path / "models" / "tfidf.joblib")
    transformed = vectorizer.transform(["a brand new sentence about a hotel room"])
    assert transformed.shape[1] == X_train.shape[1]
    assert transformed.shape[0] == 1

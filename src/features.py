"""TF-IDF features fitted on the train split only."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import AttrDict, load_config
from src.utils import ensure_dirs, get_logger, set_seed


def build_tfidf(config: AttrDict) -> TfidfVectorizer:
    """Return an unfitted TfidfVectorizer using config.tfidf settings."""
    tfidf = config.tfidf
    return TfidfVectorizer(
        ngram_range=(int(tfidf.ngram_min), int(tfidf.ngram_max)),
        max_features=int(tfidf.max_features),
        min_df=int(tfidf.min_df),
        sublinear_tf=bool(tfidf.sublinear_tf),
    )


def _load_split(config: AttrDict, name: str) -> pd.DataFrame:
    """Load one split CSV; the vectorizer is never fitted on test."""
    path = Path(config.paths.splits_dir) / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}. Run python -m src.split first.")
    return pd.read_csv(path)


def _texts_and_labels(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Return text_clean strings and integer labels."""
    texts = df["text_clean"].fillna("").astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()
    return texts, labels


def make_tfidf_features(
    config: AttrDict | None = None,
) -> tuple[spmatrix, spmatrix, spmatrix, np.ndarray, np.ndarray, np.ndarray]:
    """Fit TF-IDF on train text_clean only, then transform train, val, and test."""
    cfg = config if config is not None else load_config()
    logger = get_logger(__name__)
    set_seed(int(cfg.seed))
    ensure_dirs(cfg)
    train_df = _load_split(cfg, "train")
    val_df = _load_split(cfg, "val")
    test_df = _load_split(cfg, "test")
    train_texts, y_train = _texts_and_labels(train_df)
    val_texts, y_val = _texts_and_labels(val_df)
    test_texts, y_test = _texts_and_labels(test_df)
    vectorizer = build_tfidf(cfg)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    out_path = Path(cfg.paths.models_dir) / "tfidf.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_path)
    vocab_size = len(vectorizer.get_feature_names_out())
    logger.info("TF-IDF vocabulary size=%s", vocab_size)
    logger.info(
        "TF-IDF matrix shapes X_train=%s X_val=%s X_test=%s",
        tuple(X_train.shape),
        tuple(X_val.shape),
        tuple(X_test.shape),
    )
    print(f"vocab_size={vocab_size}")
    print(f"X_train={tuple(X_train.shape)} X_val={tuple(X_val.shape)} X_test={tuple(X_test.shape)}")
    print(f"saved={out_path}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def main() -> None:
    """Fit TF-IDF on train and transform all splits."""
    make_tfidf_features(load_config())


if __name__ == "__main__":
    main()

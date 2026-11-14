"""TF-IDF feature extraction for classical baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import load_config, parse_config_args, path_from_cfg
from src.utils import load_dataframe, save_joblib, seed_from_cfg, set_seed, setup_logging


def make_tfidf(cfg: dict[str, Any]) -> TfidfVectorizer:
    """Construct a TF-IDF vectorizer from config (unfitted)."""
    spec = cfg["features"]["tfidf"]
    ngram = spec["ngram_range"]
    return TfidfVectorizer(
        ngram_range=(int(ngram[0]), int(ngram[1])),
        max_features=int(spec["max_features"]),
        min_df=int(spec["min_df"]),
        max_df=float(spec["max_df"]),
        sublinear_tf=bool(spec["sublinear_tf"]),
    )


def fit_tfidf(texts: list[str], cfg: dict[str, Any]) -> TfidfVectorizer:
    """Fit TF-IDF on training texts."""
    vectorizer = make_tfidf(cfg)
    vectorizer.fit(texts)
    return vectorizer


def transform_tfidf(vectorizer: TfidfVectorizer, texts: list[str]) -> sparse.spmatrix:
    """Transform texts with a fitted TF-IDF vectorizer."""
    return vectorizer.transform(texts)


def texts_and_labels(df: Any, cfg: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """Extract cleaned texts and integer labels from a DataFrame."""
    text_col = cfg["columns"]["clean_text"] if cfg["columns"]["clean_text"] in df.columns else cfg["columns"]["text"]
    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[cfg["columns"]["label"]].astype(int).to_numpy()
    return texts, labels


def vectorizer_path(cfg: dict[str, Any]) -> Any:
    """Return the on-disk path for the fitted vectorizer."""
    return path_from_cfg(cfg, "paths.models") / cfg["features"]["vectorizer_filename"]


def main(argv: list[str] | None = None) -> None:
    """Fit TF-IDF on the train split and save the vectorizer."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    train = load_dataframe(path_from_cfg(cfg, "paths.splits") / "train.csv")
    texts, _ = texts_and_labels(train, cfg)
    vectorizer = fit_tfidf(texts, cfg)
    out = save_joblib(vectorizer, vectorizer_path(cfg))
    logger.info("Fitted TF-IDF with %s terms -> %s", len(vectorizer.get_feature_names_out()), out)
    print(f"wrote {out} vocab={len(vectorizer.get_feature_names_out())}")


if __name__ == "__main__":
    main()

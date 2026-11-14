"""Tests for src.features."""

from src.clean import clean_dataframe
from src.features import fit_tfidf, texts_and_labels, transform_tfidf


def test_tfidf_fit_transform_shape(cfg: dict, reviews_df) -> None:
    """TF-IDF returns a document-term matrix with one row per review."""
    cleaned = clean_dataframe(reviews_df, cfg)
    texts, labels = texts_and_labels(cleaned, cfg)
    vectorizer = fit_tfidf(texts, cfg)
    matrix = transform_tfidf(vectorizer, texts)
    assert matrix.shape[0] == len(texts)
    assert matrix.shape[1] > 0
    assert len(labels) == len(texts)
    ngram = tuple(cfg["features"]["tfidf"]["ngram_range"])
    assert vectorizer.ngram_range == ngram

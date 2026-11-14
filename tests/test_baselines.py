"""Tests for src.baselines."""

from src.baselines import baseline_names, evaluate_baseline, instantiate_estimator, train_baseline
from src.clean import clean_dataframe
from src.features import fit_tfidf, texts_and_labels, transform_tfidf


def test_instantiate_knn(cfg: dict) -> None:
    """Estimator import uses the configured sklearn class."""
    spec = cfg["baselines"]["knn"]
    model = instantiate_estimator(spec["estimator"], spec["params"], 42)
    assert model.__class__.__name__ == "KNeighborsClassifier"


def test_train_and_evaluate_decision_tree(cfg: dict, reviews_df) -> None:
    """A tiny decision tree trains and returns accuracy metrics."""
    cleaned = clean_dataframe(reviews_df, cfg)
    texts, labels = texts_and_labels(cleaned, cfg)
    vectorizer = fit_tfidf(texts, cfg)
    features = transform_tfidf(vectorizer, texts)
    model = train_baseline("decision_tree", features, labels, cfg)
    metrics = evaluate_baseline(model, "decision_tree", features, labels, cfg)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "per_class" in metrics
    assert "decision_tree" in baseline_names(cfg)

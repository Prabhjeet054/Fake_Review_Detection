"""Tests for src.experiments."""

from src.evaluate import save_metrics
from src.experiments import best_model, collect_results


def test_collect_results_and_best(cfg: dict) -> None:
    """Saved baseline and deep metrics are collected and ranked."""
    save_metrics(
        {"accuracy": 0.7, "macro_f1": 0.65, "weighted_f1": 0.66, "cohen_kappa": 0.5},
        cfg,
        "baseline_knn_test",
    )
    save_metrics(
        {"accuracy": 0.9, "macro_f1": 0.88, "weighted_f1": 0.89, "cohen_kappa": 0.8},
        cfg,
        "deep_model_test",
    )
    rows = collect_results(cfg)
    names = {row["model"] for row in rows}
    assert "knn" in names
    assert "distilbert_mlp" in names
    winner = best_model(rows, "macro_f1")
    assert winner is not None
    assert winner["model"] == "distilbert_mlp"

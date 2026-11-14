"""Train and evaluate TF-IDF + classical sklearn baselines."""

from __future__ import annotations

import inspect
import importlib
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator

from src.config import load_config, parse_config_args, path_from_cfg
from src.evaluate import compute_metrics, save_metrics
from src.features import fit_tfidf, texts_and_labels, transform_tfidf, vectorizer_path
from src.plots import plot_confusion_matrix
from src.utils import (
    load_dataframe,
    load_joblib,
    save_joblib,
    seed_from_cfg,
    set_seed,
    setup_logging,
)


def instantiate_estimator(spec: str, params: dict[str, Any], seed: int) -> BaseEstimator:
    """Import an sklearn estimator class and construct it with config params."""
    module_name, class_name = spec.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    kwargs = dict(params)
    signature = inspect.signature(cls.__init__)
    if "random_state" in signature.parameters:
        kwargs.setdefault("random_state", seed)
    return cls(**kwargs)


def baseline_names(cfg: dict[str, Any]) -> list[str]:
    """Return configured baseline model names in YAML order."""
    return list(cfg["baselines"].keys())


def maybe_dense(matrix: sparse.spmatrix | np.ndarray, estimator_name: str) -> sparse.spmatrix | np.ndarray:
    """Convert to dense arrays for estimators that do not accept sparse input."""
    dense_models = {"gradient_boosting", "adaboost"}
    if estimator_name in dense_models and sparse.issparse(matrix):
        return matrix.toarray()
    return matrix


def train_baseline(
    name: str,
    x_train: sparse.spmatrix | np.ndarray,
    y_train: np.ndarray,
    cfg: dict[str, Any],
) -> BaseEstimator:
    """Fit a single named baseline estimator."""
    spec = cfg["baselines"][name]
    model = instantiate_estimator(spec["estimator"], spec.get("params", {}), seed_from_cfg(cfg))
    model.fit(maybe_dense(x_train, name), y_train)
    return model


def evaluate_baseline(
    model: BaseEstimator,
    name: str,
    x: sparse.spmatrix | np.ndarray,
    y: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Predict and compute metrics for one baseline."""
    preds = np.asarray(model.predict(maybe_dense(x, name)))
    metrics = compute_metrics(y, preds, cfg)
    metrics["model"] = name
    return metrics


def model_path(cfg: dict[str, Any], name: str) -> Any:
    """Path to a saved baseline model."""
    return path_from_cfg(cfg, "paths.models") / f"baseline_{name}.joblib"


def main(argv: list[str] | None = None) -> None:
    """Train every baseline on train TF-IDF and evaluate on val/test."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    split_dir = path_from_cfg(cfg, "paths.splits")
    train_df = load_dataframe(split_dir / "train.csv")
    val_df = load_dataframe(split_dir / "val.csv")
    test_df = load_dataframe(split_dir / "test.csv")
    train_texts, y_train = texts_and_labels(train_df, cfg)
    val_texts, y_val = texts_and_labels(val_df, cfg)
    test_texts, y_test = texts_and_labels(test_df, cfg)
    stored = vectorizer_path(cfg)
    if stored.exists():
        vectorizer = load_joblib(stored)
    else:
        vectorizer = fit_tfidf(train_texts, cfg)
        save_joblib(vectorizer, stored)
    x_train = transform_tfidf(vectorizer, train_texts)
    x_val = transform_tfidf(vectorizer, val_texts)
    x_test = transform_tfidf(vectorizer, test_texts)
    summary: list[dict[str, Any]] = []
    for name in baseline_names(cfg):
        logger.info("Training baseline %s", name)
        model = train_baseline(name, x_train, y_train, cfg)
        save_joblib(model, model_path(cfg, name))
        val_metrics = evaluate_baseline(model, name, x_val, y_val, cfg)
        test_metrics = evaluate_baseline(model, name, x_test, y_test, cfg)
        save_metrics(val_metrics, cfg, f"baseline_{name}_val")
        save_metrics(test_metrics, cfg, f"baseline_{name}_test")
        plot_confusion_matrix(
            np.array(test_metrics["confusion_matrix"]),
            cfg,
            stem=f"cm_baseline_{name}_test",
            title=f"{name} test confusion matrix",
        )
        row = {"model": name, "val_accuracy": val_metrics["accuracy"], "test_accuracy": test_metrics["accuracy"]}
        row["macro_f1"] = test_metrics["macro_f1"]
        row["weighted_f1"] = test_metrics["weighted_f1"]
        row["cohen_kappa"] = test_metrics["cohen_kappa"]
        summary.append(row)
        logger.info("%s test acc=%.4f macro_f1=%.4f", name, test_metrics["accuracy"], test_metrics["macro_f1"])
    from src.utils import save_json

    save_json(summary, path_from_cfg(cfg, "paths.metrics") / "baselines_summary.json")
    print(f"trained {len(summary)} baselines")


if __name__ == "__main__":
    main()

"""Train TF-IDF baselines and evaluate them on val and test."""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.config import AttrDict, load_config
from src.features import make_tfidf_features
from src.utils import ensure_dirs, get_logger, set_seed

DENSE_MODELS = frozenset({"adaboost", "gradient_boosting"})
MODEL_ORDER = (
    "majority_class",
    "knn",
    "decision_tree",
    "random_forest",
    "svm",
    "adaboost",
    "gradient_boosting",
)
DISPLAY_NAMES = {
    "majority_class": "Majority class",
    "knn": "KNN",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "adaboost": "AdaBoost",
    "gradient_boosting": "Gradient Boosting",
}


def _with_seed(cls: type, seed: int, **kwargs: Any) -> Any:
    """Construct an estimator, passing random_state when the class supports it."""
    if "random_state" in inspect.signature(cls.__init__).parameters:
        kwargs["random_state"] = seed
    return cls(**kwargs)


def build_estimators(seed: int) -> dict[str, Any]:
    """Return the configured baseline estimators, seeded where supported."""
    return {
        "majority_class": _with_seed(DummyClassifier, seed, strategy="most_frequent"),
        "knn": KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
        "decision_tree": _with_seed(DecisionTreeClassifier, seed),
        "random_forest": _with_seed(RandomForestClassifier, seed, n_estimators=300, n_jobs=-1),
        "svm": _with_seed(LinearSVC, seed),
        "adaboost": _with_seed(AdaBoostClassifier, seed, n_estimators=200),
        "gradient_boosting": _with_seed(GradientBoostingClassifier, seed, n_estimators=200),
    }


def _as_matrix(name: str, matrix: Any) -> Any:
    """Return a dense array for models that do not accept sparse input."""
    if name in DENSE_MODELS and hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy and macro/weighted precision, recall, and F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _split_ids(config: AttrDict, name: str) -> np.ndarray:
    """Load row ids for a split CSV."""
    path = Path(config.paths.splits_dir) / f"{name}.csv"
    return pd.read_csv(path, usecols=["id"])["id"].to_numpy()


def train_baselines(config: AttrDict | None = None) -> pd.DataFrame:
    """Fit every baseline on train TF-IDF and evaluate on val and test."""
    cfg = config if config is not None else load_config()
    logger = get_logger(__name__)
    set_seed(int(cfg.seed))
    ensure_dirs(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = make_tfidf_features(cfg)
    splits = {
        "val": (X_val, y_val, _split_ids(cfg, "val")),
        "test": (X_test, y_test, _split_ids(cfg, "test")),
    }
    rows: list[dict] = []
    preds_by_split: dict[str, dict[str, np.ndarray]] = {
        "val": {"id": splits["val"][2], "y_true": y_val},
        "test": {"id": splits["test"][2], "y_true": y_test},
    }
    estimators = build_estimators(int(cfg.seed))
    models_dir = Path(cfg.paths.models_dir)
    metrics_dir = Path(cfg.paths.metrics_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for name in MODEL_ORDER:
        model = estimators[name]
        x_fit = _as_matrix(name, X_train)
        started = time.perf_counter()
        model.fit(x_fit, y_train)
        elapsed = time.perf_counter() - started
        logger.info("Trained %s in %.2fs", DISPLAY_NAMES[name], elapsed)
        joblib.dump(model, models_dir / f"baseline_{name}.joblib")
        for split_name, (x_split, y_split, _ids) in splits.items():
            y_pred = np.asarray(model.predict(_as_matrix(name, x_split)))
            preds_by_split[split_name][name] = y_pred
            metrics = classification_metrics(y_split, y_pred)
            rows.append(
                {
                    "model": DISPLAY_NAMES[name],
                    "model_key": name,
                    "split": split_name,
                    "train_seconds": elapsed,
                    **metrics,
                }
            )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metrics_dir / "baselines.csv", index=False)
    for split_name, payload in preds_by_split.items():
        frame = pd.DataFrame(
            {
                "id": payload["id"],
                "y_true": payload["y_true"],
                **{key: payload[key] for key in MODEL_ORDER},
            }
        )
        frame.to_csv(metrics_dir / f"baseline_predictions_{split_name}.csv", index=False)
    test_table = (
        metrics_df.loc[metrics_df["split"] == "test"]
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    print("\nBaselines sorted by test macro F1:")
    print(
        test_table[
            ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "train_seconds"]
        ].to_string(index=False)
    )
    return metrics_df


def main() -> None:
    """Train baselines from config.yaml and write metrics and models."""
    train_baselines(load_config())


if __name__ == "__main__":
    main()

# This is the only DistilBERT code that loads the test split.
"""Evaluate the saved ClassifierHead on held-out test embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn

from src.config import AttrDict, load_config
from src.embed import embedding_paths
from src.model import ClassifierHead
from src.schema import ID2LABEL
from src.utils import ensure_dirs, get_device, get_logger, save_json, set_seed

PRED_COLUMNS = ("id", "text_clean", "true", "pred", "p0", "p1", "p2")


def class_names(config: AttrDict) -> list[str]:
    """Return class names in id order 0..num_classes-1."""
    n_classes = int(config.model.num_classes)
    return [str(ID2LABEL[index]) for index in range(n_classes)]


def load_best_head(config: AttrDict, device: str) -> nn.Module:
    """Load outputs/models/head_best.pt onto device in eval mode."""
    path = Path(config.paths.models_dir) / "head_best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run python -m src.train first.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ClassifierHead(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def load_test_embeddings(config: AttrDict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load test X, y, and ids npy caches (the only DistilBERT test-split read)."""
    paths = embedding_paths(config, "test")
    for key in ("X", "y", "ids"):
        if not paths[key].is_file():
            raise FileNotFoundError(
                f"Missing embedding file: {paths[key]}. Run python -m src.embed first."
            )
    X = np.load(paths["X"]).astype(np.float32, copy=False)
    y = np.load(paths["y"]).astype(np.int64, copy=False)
    ids = np.load(paths["ids"])
    return X, y, ids


def load_test_texts(config: AttrDict, ids: np.ndarray) -> list[str]:
    """Return text_clean for embedding ids, in the same order as the npy cache."""
    csv_path = Path(config.paths.splits_dir) / "test.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing split file: {csv_path}. Run python -m src.split first.")
    frame = pd.read_csv(csv_path, usecols=["id", "text_clean"])
    by_id = {
        int(key): str(text)
        for key, text in zip(frame["id"].to_numpy(), frame["text_clean"].fillna("").astype(str))
    }
    missing = [item for item in ids if int(item) not in by_id]
    if missing:
        raise KeyError(f"test.csv is missing {len(missing)} embedding id(s), e.g. {missing[0]!r}.")
    return [by_id[int(item)] for item in ids]


@torch.no_grad()
def predict_proba_and_labels(
    model: nn.Module,
    X: np.ndarray,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return softmax probabilities (n, C) and argmax labels (n,) in eval mode."""
    model.eval()
    chunks: list[np.ndarray] = []
    size = int(batch_size)
    for start in range(0, len(X), size):
        batch = torch.from_numpy(np.asarray(X[start : start + size], dtype=np.float32)).to(device)
        logits = model(batch)
        chunks.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64))
    if not chunks:
        n_classes = int(model.fc2.out_features)
        return np.zeros((0, n_classes), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    probs = np.vstack(chunks)
    labels = probs.argmax(axis=1).astype(np.int64)
    return probs, labels


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: AttrDict,
) -> dict[str, Any]:
    """Accuracy, per-class scores, macro/weighted averages, kappa, and confusion matrix."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(int(config.model.num_classes)))
    names = class_names(config)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = {
        names[index]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in labels
    }
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    n_test = len(y_true)
    n_correct = int((y_true == y_pred).sum())
    return {
        "n_test": n_test,
        "n_correct": n_correct,
        "accuracy": float(n_correct / n_test) if n_test else 0.0,
        "per_class": per_class,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_precision": float(np.average(precision, weights=support) if n_test else 0.0),
        "weighted_recall": float(np.average(recall, weights=support) if n_test else 0.0),
        "weighted_f1": float(np.average(f1, weights=support) if n_test else 0.0),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)) if n_test else 0.0,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def write_predictions(
    ids: np.ndarray,
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    path: Path,
) -> Path:
    """Write id, text_clean, true, pred, p0, p1, p2."""
    frame = pd.DataFrame(
        {
            "id": ids,
            "text_clean": texts,
            "true": np.asarray(y_true, dtype=np.int64),
            "pred": np.asarray(y_pred, dtype=np.int64),
            "p0": probs[:, 0],
            "p1": probs[:, 1],
            "p2": probs[:, 2],
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def evaluate_test(config: AttrDict | None = None) -> dict[str, Any]:
    """Score head_best.pt on test embeddings and write CSV plus metrics.json."""
    cfg = config if config is not None else load_config()
    set_seed(int(cfg.seed))
    ensure_dirs(cfg)
    logger = get_logger(__name__)
    device = get_device()
    model = load_best_head(cfg, device)
    X, y_true, ids = load_test_embeddings(cfg)
    texts = load_test_texts(cfg, ids)
    probs, y_pred = predict_proba_and_labels(model, X, device, int(cfg.training.batch_size))
    metrics = compute_metrics(y_true, y_pred, cfg)
    pred_path = Path(cfg.paths.metrics_dir) / "test_predictions.csv"
    metrics_path = Path(cfg.paths.metrics_dir) / "metrics.json"
    write_predictions(ids, texts, y_true, y_pred, probs, pred_path)
    save_json(metrics, metrics_path)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(int(cfg.model.num_classes))),
        target_names=class_names(cfg),
        zero_division=0,
        digits=4,
    )
    print(report)
    logger.info(
        "test n=%s correct=%s accuracy=%.4f macro_f1=%.4f kappa=%.4f -> %s",
        metrics["n_test"],
        metrics["n_correct"],
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["cohen_kappa"],
        metrics_path,
    )
    print(f"wrote {pred_path}")
    print(f"wrote {metrics_path}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    """CLI: evaluate the best classification head on the test split."""
    parser = argparse.ArgumentParser(description="Evaluate ClassifierHead on test embeddings.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args(argv)
    evaluate_test(load_config(args.config))


if __name__ == "__main__":
    main()

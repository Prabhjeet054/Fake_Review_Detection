"""Classification metrics, including grouped binary evaluations."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import label_names, load_config, parse_config_args, path_from_cfg
from src.embed import load_embeddings
from src.model import load_classifier
from src.plots import plot_confusion_matrix
from src.utils import get_device, save_json, seed_from_cfg, set_seed, setup_logging


def _per_class_table(y_true: np.ndarray, y_pred: np.ndarray, cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Precision/recall/F1 for each configured class."""
    labels = list(range(int(cfg["model"]["num_classes"])))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    names = label_names(cfg)
    table: dict[str, dict[str, float]] = {}
    for index, class_id in enumerate(labels):
        table[names[class_id]] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    return table


def grouped_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Collapse the 3-class problem into configured binary groupings."""
    results: dict[str, dict[str, float]] = {}
    for name, spec in cfg["evaluation"]["grouped_binary"].items():
        positive = {int(v) for v in spec["positive"]}
        negative = {int(v) for v in spec["negative"]}
        exclude = {int(v) for v in spec.get("exclude", [])}
        mask = np.ones(len(y_true), dtype=bool)
        if exclude:
            mask &= ~np.isin(y_true, list(exclude))
            mask &= ~np.isin(y_pred, list(exclude))
        keep_true = y_true[mask]
        keep_pred = y_pred[mask]
        if keep_true.size == 0:
            results[name] = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
            continue
        bin_true = np.isin(keep_true, list(positive)).astype(int)
        bin_pred = np.isin(keep_pred, list(positive)).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            bin_true,
            bin_pred,
            average="binary",
            zero_division=0,
        )
        results[name] = {
            "accuracy": float(accuracy_score(bin_true, bin_pred)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(len(bin_true)),
            "positive_labels": sorted(positive),
            "negative_labels": sorted(negative),
        }
    return results


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute accuracy, per-class scores, averages, kappa, and confusion matrix."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(int(cfg["model"]["num_classes"])))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[label_names(cfg)[i] for i in labels],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": _per_class_table(y_true, y_pred, cfg),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "grouped_binary": grouped_binary_metrics(y_true, y_pred, cfg),
        "n": int(len(y_true)),
    }
    return metrics


def save_metrics(metrics: dict[str, Any], cfg: dict[str, Any], stem: str) -> Any:
    """Write a metrics JSON file under the configured metrics directory."""
    return save_json(metrics, path_from_cfg(cfg, "paths.metrics") / f"{stem}.json")


def predict_classifier(embeddings: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Load the trained head and predict class ids for embedding rows."""
    import torch

    device = get_device(cfg)
    model = load_classifier(cfg, map_location=device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(device)
        logits = model(tensor)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
    return preds.astype(int)


def main(argv: list[str] | None = None) -> None:
    """Evaluate the trained deep model on the test embeddings."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    embeddings, labels, _ = load_embeddings(cfg, "test")
    preds = predict_classifier(embeddings, cfg)
    metrics = compute_metrics(labels, preds, cfg)
    path = save_metrics(metrics, cfg, "deep_model_test")
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        cfg,
        stem="cm_deep_model_test",
        title="Deep model test confusion matrix",
    )
    logger.info("Deep model test accuracy=%.4f macro_f1=%.4f", metrics["accuracy"], metrics["macro_f1"])
    print(f"wrote {path} accuracy={metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()

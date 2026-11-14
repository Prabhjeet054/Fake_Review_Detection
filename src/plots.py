"""Plotting helpers for EDA, metrics, and training curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import label_names, load_config, parse_config_args, path_from_cfg
from src.utils import ensure_dir, seed_from_cfg, set_seed


def _style(cfg: dict[str, Any]) -> None:
    """Apply seaborn style from config."""
    sns.set_theme(style=cfg["plots"]["style"], palette=cfg["plots"]["palette"])


def figure_path(cfg: dict[str, Any], stem: str) -> Path:
    """Return the output path for a named figure."""
    directory = ensure_dir(path_from_cfg(cfg, "paths.figures"))
    return directory / f"{stem}.{cfg['plots']['format']}"


def save_current_fig(cfg: dict[str, Any], stem: str) -> Path:
    """Save the active matplotlib figure using plot settings from config."""
    path = figure_path(cfg, stem)
    plt.tight_layout()
    plt.savefig(path, dpi=int(cfg["plots"]["dpi"]))
    plt.close()
    return path


def plot_class_distribution(df: pd.DataFrame, cfg: dict[str, Any], stem: str = "class_distribution") -> Path:
    """Bar chart of class counts."""
    _style(cfg)
    names = label_names(cfg)
    label_col = cfg["columns"]["label"]
    counts = df[label_col].value_counts().sort_index()
    labels = [names.get(int(idx), str(idx)) for idx in counts.index]
    figsize = tuple(cfg["plots"]["figsize"])
    plt.figure(figsize=figsize)
    sns.barplot(x=labels, y=counts.values)
    plt.title("Class distribution", fontsize=cfg["plots"]["title_fontsize"])
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    return save_current_fig(cfg, stem)


def plot_length_distribution(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    text_col: str,
    stem: str = "length_distribution",
) -> Path:
    """Histogram of token lengths by class."""
    _style(cfg)
    names = label_names(cfg)
    work = df.copy()
    work["n_tokens"] = work[text_col].fillna("").astype(str).str.split().map(len)
    work["class_name"] = work[cfg["columns"]["label"]].map(lambda value: names.get(int(value), str(value)))
    plt.figure(figsize=tuple(cfg["plots"]["figsize"]))
    sns.histplot(data=work, x="n_tokens", hue="class_name", element="step", stat="density", common_norm=False)
    plt.title("Review length by class", fontsize=cfg["plots"]["title_fontsize"])
    plt.xlabel("Tokens")
    return save_current_fig(cfg, stem)


def plot_confusion_matrix(
    matrix: np.ndarray,
    cfg: dict[str, Any],
    stem: str = "confusion_matrix",
    title: str = "Confusion matrix",
) -> Path:
    """Heatmap of a square confusion matrix."""
    _style(cfg)
    names = [label_names(cfg)[i] for i in range(int(cfg["model"]["num_classes"]))]
    plt.figure(figsize=tuple(cfg["plots"]["figsize"]))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(title, fontsize=cfg["plots"]["title_fontsize"])
    plt.xticks(rotation=20)
    plt.yticks(rotation=0)
    return save_current_fig(cfg, stem)


def plot_metrics_comparison(rows: list[dict[str, Any]], cfg: dict[str, Any], stem: str = "metrics_comparison") -> Path:
    """Grouped bar chart comparing models on a selected metric."""
    _style(cfg)
    frame = pd.DataFrame(rows)
    metric = cfg["experiments"]["metric"]
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' is not in comparison rows.")
    plt.figure(figsize=tuple(cfg["plots"]["figsize"]))
    sns.barplot(data=frame, x="model", y=metric)
    plt.title(f"Model comparison ({metric})", fontsize=cfg["plots"]["title_fontsize"])
    plt.xticks(rotation=25)
    plt.ylabel(metric)
    return save_current_fig(cfg, stem)


def plot_training_curves(history: pd.DataFrame, cfg: dict[str, Any], stem: str = "training_curves") -> Path:
    """Plot train/val loss and accuracy over epochs."""
    _style(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(cfg["plots"]["figsize"][0] * 1.6, cfg["plots"]["figsize"][1]))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    if "val_loss" in history.columns:
        axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_acc"], label="train")
    if "val_acc" in history.columns:
        axes[1].plot(history["epoch"], history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle("Training curves", fontsize=cfg["plots"]["title_fontsize"])
    path = figure_path(cfg, stem)
    fig.tight_layout()
    fig.savefig(path, dpi=int(cfg["plots"]["dpi"]))
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    """Write a placeholder class-distribution figure from split data if available."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    split_dir = path_from_cfg(cfg, "paths.splits")
    train_path = split_dir / "train.csv"
    if not train_path.exists():
        print("No train split found; plots module imported successfully.")
        return
    from src.utils import load_dataframe

    df = load_dataframe(train_path)
    path = plot_class_distribution(df, cfg, "plots_module_class_distribution")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

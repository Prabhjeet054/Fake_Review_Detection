"""Baseline bar charts and training history curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from src.config import AttrDict, load_config
from src.schema import ID2LABEL
from src.utils import ensure_dirs, load_json

CURVE_FIGSIZE = (8, 5)
CURVE_DPI = 200
TRAIN_COLOR = "#4C78A8"
VAL_COLOR = "#E45756"
CURVE_FONT = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

CLASSIFIER_ORDER = ("KNN", "RF", "DT", "SVM", "AB", "GB")
SHORT_NAMES = {
    "KNN": "KNN",
    "Random Forest": "RF",
    "Decision Tree": "DT",
    "SVM": "SVM",
    "AdaBoost": "AB",
    "Gradient Boosting": "GB",
}
METRIC_SPECS = (
    ("accuracy", "Accuracy"),
    ("macro_precision", "Precision"),
    ("macro_recall", "Recall"),
    ("macro_f1", "F1"),
)
BAR_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
PER_CLASS_SPECS = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
)


def _split_rows(split: str, config: AttrDict | None = None) -> pd.DataFrame:
    """Load baselines.csv for one split, excluding the majority-class row."""
    cfg = config if config is not None else load_config()
    csv_path = Path(cfg.paths.metrics_dir) / "baselines.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}. Run python -m src.baselines first.")
    df = pd.read_csv(csv_path)
    df = df.loc[df["model"].str.lower() != "majority class"].copy()
    df = df.loc[df["split"].astype(str).str.lower() == split.lower()].copy()
    df["short"] = df["model"].map(SHORT_NAMES)
    df = df.loc[df["short"].isin(CLASSIFIER_ORDER)].copy()
    df["short"] = pd.Categorical(df["short"], categories=CLASSIFIER_ORDER, ordered=True)
    df = df.sort_values("short")
    if df.empty:
        raise ValueError(f"No baseline rows found for split '{split}'.")
    return df


def build_baselines_figure(split: str, config: AttrDict | None = None) -> tuple[Any, Any]:
    """Rebuild the grouped bar chart and return (figure, axes) without saving."""
    df = _split_rows(split, config)
    x = np.arange(len(CLASSIFIER_ORDER))
    width = 0.18
    offsets = np.linspace(-1.5, 1.5, num=len(METRIC_SPECS)) * width
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, color, (column, label) in zip(offsets, BAR_COLORS, METRIC_SPECS):
        values = []
        for name in CLASSIFIER_ORDER:
            row = df.loc[df["short"] == name]
            values.append(100.0 * float(row[column].iloc[0]) if len(row) else 0.0)
        bars = ax.bar(x + offset, values, width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                clip_on=False,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(list(CLASSIFIER_ORDER))
    ax.set_xlabel("Classifiers")
    ax.set_ylabel("Evaluation metrics (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4, frameon=True)
    ax.set_title(f"Baseline metrics ({split})")
    fig.tight_layout()
    return fig, ax


def plot_baselines(split: str, config: AttrDict | None = None) -> Path:
    """Draw a grouped bar chart of baseline metrics for one split and save it at 200 dpi."""
    cfg = config if config is not None else load_config()
    ensure_dirs(cfg)
    fig, _ax = build_baselines_figure(split, cfg)
    out_path = Path(cfg.paths.figures_dir) / f"baselines_{split}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def load_history(config: AttrDict | None = None) -> pd.DataFrame:
    """Load outputs/metrics/history.csv."""
    cfg = config if config is not None else load_config()
    path = Path(cfg.paths.metrics_dir) / "history.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run python -m src.train first.")
    return pd.read_csv(path)


def best_epoch_from_history(history: pd.DataFrame) -> int:
    """Return the epoch with the lowest validation loss."""
    return int(history.loc[history["val_loss"].idxmin(), "epoch"])


def _apply_curve_style(ax: Any, title: str, ylabel: str, best_epoch: int, n_epochs: int) -> None:
    """Apply the shared curve style: title, labels, 1..N x-axis, light grid, best-epoch line."""
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, n_epochs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.axvline(best_epoch, color="#555555", linestyle=":", linewidth=1.6, zorder=1)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, color="#d9d9d9")
    ax.set_axisbelow(True)
    ax.legend(frameon=True)


def build_history_curve(kind: str, history: pd.DataFrame) -> tuple[Any, Any]:
    """Build an accuracy or loss curve figure; kind is 'accuracy' or 'loss'."""
    if kind not in {"accuracy", "loss"}:
        raise ValueError(f"kind must be 'accuracy' or 'loss', got {kind!r}")
    n_epochs = int(history["epoch"].max())
    best_epoch = best_epoch_from_history(history)
    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)
    x = history["epoch"].to_numpy()
    if kind == "accuracy":
        ax.plot(x, history["train_acc"], color=TRAIN_COLOR, linewidth=2.0, label="Training Accuracy")
        ax.plot(x, history["val_acc"], color=VAL_COLOR, linewidth=2.0, label="Validation Accuracy")
        _apply_curve_style(ax, "Training and validation accuracy", "Accuracy", best_epoch, n_epochs)
    else:
        ax.plot(x, history["train_loss"], color=TRAIN_COLOR, linewidth=2.0, label="Training Loss")
        ax.plot(x, history["val_loss"], color=VAL_COLOR, linewidth=2.0, label="Validation Loss")
        _apply_curve_style(ax, "Training and validation loss", "Loss", best_epoch, n_epochs)
    fig.tight_layout()
    return fig, ax


def plot_history(config: AttrDict | None = None) -> dict[str, Path]:
    """Read history.csv and save accuracy_curve.png and loss_curve.png."""
    cfg = config if config is not None else load_config()
    ensure_dirs(cfg)
    history = load_history(cfg)
    plt.rcParams.update(CURVE_FONT)
    figures_dir = Path(cfg.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "accuracy": figures_dir / "accuracy_curve.png",
        "loss": figures_dir / "loss_curve.png",
    }
    for kind, out_path in paths.items():
        fig, _ax = build_history_curve(kind, history)
        fig.savefig(out_path, dpi=CURVE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")
    return paths


def load_metrics(config: AttrDict | None = None) -> dict[str, Any]:
    """Load outputs/metrics/metrics.json from the test-set evaluation."""
    cfg = config if config is not None else load_config()
    path = Path(cfg.paths.metrics_dir) / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run python -m src.evaluate first.")
    return load_json(path)


def class_labels(config: AttrDict | None = None) -> list[str]:
    """Return class names in id order for confusion-matrix axes."""
    cfg = config if config is not None else load_config()
    return [str(ID2LABEL[index]) for index in range(int(cfg.model.num_classes))]


def confusion_arrays(config: AttrDict | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return count matrix, row-normalised percent matrix, and class names."""
    cfg = config if config is not None else load_config()
    metrics = load_metrics(cfg)
    names = class_labels(cfg)
    counts = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    row_sums = counts.sum(axis=1, keepdims=True).astype(np.float64)
    normalized = np.divide(counts, row_sums, out=np.zeros((counts.shape[0], counts.shape[1]), dtype=np.float64), where=row_sums != 0) * 100.0
    return counts, normalized, names


def build_confusion_heatmap(normalized: bool, config: AttrDict | None = None) -> tuple[Any, Any]:
    """Build a seaborn heatmap of counts or row-normalised percents."""
    counts, percents, names = confusion_arrays(config)
    data = percents if normalized else counts
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        data,
        annot=True,
        fmt=".1f" if normalized else "d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        square=True,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised %)" if normalized else "Confusion matrix (counts)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return fig, ax


def plot_confusion_matrix(config: AttrDict | None = None) -> dict[str, Path]:
    """Save count and row-normalised test confusion-matrix heatmaps."""
    cfg = config if config is not None else load_config()
    ensure_dirs(cfg)
    plt.rcParams.update(CURVE_FONT)
    figures_dir = Path(cfg.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "counts": figures_dir / "confusion_matrix.png",
        "normalized": figures_dir / "confusion_matrix_normalized.png",
    }
    for key, normalized in (("counts", False), ("normalized", True)):
        fig, _ax = build_confusion_heatmap(normalized=normalized, config=cfg)
        fig.savefig(paths[key], dpi=CURVE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {paths[key]}")
    return paths


def build_per_class_metrics_figure(config: AttrDict | None = None) -> tuple[Any, Any]:
    """Build a grouped bar chart of precision, recall, and F1 per class."""
    cfg = config if config is not None else load_config()
    metrics = load_metrics(cfg)
    names = class_labels(cfg)
    x = np.arange(len(names))
    width = 0.24
    offsets = np.linspace(-1, 1, num=len(PER_CLASS_SPECS)) * width
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, color, (column, label) in zip(offsets, BAR_COLORS, PER_CLASS_SPECS):
        values = [100.0 * float(metrics["per_class"][name][column]) for name in names]
        bars = ax.bar(x + offset, values, width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=False,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-class precision, recall, and F1")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=True)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, color="#d9d9d9")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, ax


def plot_per_class_metrics(config: AttrDict | None = None) -> Path:
    """Save a grouped bar chart of per-class precision, recall, and F1."""
    cfg = config if config is not None else load_config()
    ensure_dirs(cfg)
    plt.rcParams.update(CURVE_FONT)
    fig, _ax = build_per_class_metrics_figure(cfg)
    out_path = Path(cfg.paths.figures_dir) / "per_class_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=CURVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    """Write baseline, history, confusion-matrix, and per-class figures."""
    plot_baselines("val")
    plot_baselines("test")
    plot_history()
    plot_confusion_matrix()
    plot_per_class_metrics()


if __name__ == "__main__":
    main()

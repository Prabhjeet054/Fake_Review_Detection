"""Grouped bar charts for classical baseline metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import AttrDict, load_config
from src.utils import ensure_dirs

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


def build_baselines_figure(split: str, config: AttrDict | None = None):
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


def main() -> None:
    """Write val and test baseline comparison figures."""
    plot_baselines("val")
    plot_baselines("test")


if __name__ == "__main__":
    main()

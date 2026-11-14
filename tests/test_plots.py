"""Tests for baseline metric bar charts against baselines.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from src.config import load_config
from src.plots import (
    CLASSIFIER_ORDER,
    CURVE_FIGSIZE,
    METRIC_SPECS,
    PER_CLASS_SPECS,
    SHORT_NAMES,
    best_epoch_from_history,
    build_baselines_figure,
    build_confusion_heatmap,
    build_history_curve,
    build_per_class_metrics_figure,
    class_labels,
    confusion_arrays,
    load_history,
    load_metrics,
    plot_baselines,
    plot_confusion_matrix,
    plot_history,
    plot_per_class_metrics,
)


def test_plot_baselines_writes_val_and_test_pngs(tmp_path: Path) -> None:
    """Val and test figures are written at the configured figures directory."""
    cfg = load_config()
    cfg.paths.figures_dir = tmp_path
    val_path = plot_baselines("val", cfg)
    test_path = plot_baselines("test", cfg)
    assert val_path == tmp_path / "baselines_val.png"
    assert test_path == tmp_path / "baselines_test.png"
    assert val_path.is_file() and val_path.stat().st_size > 10_240
    assert test_path.is_file() and test_path.stat().st_size > 10_240


def test_bar_heights_match_csv_and_yaxis_label() -> None:
    """Bar heights match baselines.csv x 100 and the y-axis label is exact."""
    cfg = load_config()
    csv = pd.read_csv(Path(cfg.paths.metrics_dir) / "baselines.csv")
    csv = csv.loc[csv["model"].str.lower() != "majority class"].copy()
    csv["short"] = csv["model"].map(SHORT_NAMES)
    for split in ("val", "test"):
        fig, ax = build_baselines_figure(split, cfg)
        assert ax.get_ylabel() == "Evaluation metrics (%)"
        bars = [patch for patch in ax.patches if isinstance(patch, Rectangle) and patch.get_width() > 0]
        expected: list[float] = []
        subset = csv.loc[csv["split"].astype(str).str.lower() == split]
        for column, _label in METRIC_SPECS:
            for name in CLASSIFIER_ORDER:
                value = float(subset.loc[subset["short"] == name, column].iloc[0])
                expected.append(100.0 * value)
        assert len(bars) == len(expected) == 24
        for bar, value in zip(bars, expected):
            assert bar.get_height() == pytest.approx(value, abs=0.01)
        plt.close(fig)


def test_plot_history_writes_pngs_and_shared_style(tmp_path: Path) -> None:
    """Accuracy and loss curves share style, exact legends, and a best-epoch dotted line."""
    cfg = load_config()
    cfg.paths.figures_dir = tmp_path
    paths = plot_history(cfg)
    assert paths["accuracy"] == tmp_path / "accuracy_curve.png"
    assert paths["loss"] == tmp_path / "loss_curve.png"
    assert paths["accuracy"].is_file() and paths["accuracy"].stat().st_size > 10_240
    assert paths["loss"].is_file() and paths["loss"].stat().st_size > 10_240

    history = load_history(cfg)
    best_epoch = best_epoch_from_history(history)
    n_epochs = int(history["epoch"].max())
    expected_labels = {
        "accuracy": ["Training Accuracy", "Validation Accuracy"],
        "loss": ["Training Loss", "Validation Loss"],
    }
    expected_titles = {
        "accuracy": "Training and validation accuracy",
        "loss": "Training and validation loss",
    }
    figures = {}
    for kind in ("accuracy", "loss"):
        fig, ax = build_history_curve(kind, history)
        figures[kind] = (fig, ax)
        assert ax.get_title() == expected_titles[kind]
        assert ax.get_xlabel() == "Epoch"
        assert ax.get_xlim() == (1.0, float(n_epochs))
        handles, labels = ax.get_legend_handles_labels()
        assert labels == expected_labels[kind]
        dotted = [line for line in ax.lines if line.get_linestyle() == ":"]
        assert len(dotted) == 1
        assert dotted[0].get_xdata()[0] == pytest.approx(best_epoch)
        assert any(line.get_visible() for line in ax.get_xgridlines())
        assert tuple(fig.get_size_inches()) == CURVE_FIGSIZE
    acc_ax = figures["accuracy"][1]
    loss_ax = figures["loss"][1]
    assert acc_ax.xaxis.label.get_fontname() == loss_ax.xaxis.label.get_fontname()
    assert acc_ax.xaxis.label.get_fontsize() == loss_ax.xaxis.label.get_fontsize()
    assert acc_ax.title.get_fontsize() == loss_ax.title.get_fontsize()
    plt.close(figures["accuracy"][0])
    plt.close(figures["loss"][0])


def test_history_curve_line_data_matches_csv_and_xaxis_is_1_to_50() -> None:
    """Rebuilt curve x/y match history.csv exactly; both x-axes run from epoch 1 to 50."""
    history = load_history()
    assert list(history["epoch"].astype(int)) == list(range(1, 51))
    expected = {
        "accuracy": (
            ("Training Accuracy", "train_acc"),
            ("Validation Accuracy", "val_acc"),
        ),
        "loss": (
            ("Training Loss", "train_loss"),
            ("Validation Loss", "val_loss"),
        ),
    }
    epochs = history["epoch"].to_numpy(dtype=np.float64)
    for kind, series in expected.items():
        fig, ax = build_history_curve(kind, history)
        assert ax.get_xlim() == (1.0, 50.0)
        labeled = [line for line in ax.lines if line.get_label() in {name for name, _col in series}]
        assert len(labeled) == 2
        by_label = {line.get_label(): line for line in labeled}
        for name, column in series:
            line = by_label[name]
            np.testing.assert_allclose(np.asarray(line.get_xdata(), dtype=np.float64), epochs, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(
                np.asarray(line.get_ydata(), dtype=np.float64),
                history[column].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=0.0,
            )
        plt.close(fig)


def test_heatmap_annotations_match_confusion_matrix_and_rows_sum_to_100() -> None:
    """Count-cell texts equal metrics.json; each normalised row sums to 100%."""
    cfg = load_config()
    metrics = load_metrics(cfg)
    expected_cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    fig, ax = build_confusion_heatmap(normalized=False, config=cfg)
    count_annot = np.array([int(text.get_text()) for text in ax.texts], dtype=int).reshape(expected_cm.shape)
    np.testing.assert_array_equal(count_annot, expected_cm)
    plt.close(fig)
    _counts, percents, _names = confusion_arrays(cfg)
    np.testing.assert_allclose(percents.sum(axis=1), np.full(expected_cm.shape[0], 100.0), atol=1e-9)


def test_per_class_bar_heights_equal_metrics_json() -> None:
    """Bar heights equal per-class precision, recall, and F1 in metrics.json (as percent)."""
    cfg = load_config()
    metrics = load_metrics(cfg)
    names = class_labels(cfg)
    fig, ax = build_per_class_metrics_figure(cfg)
    bars = [patch for patch in ax.patches if isinstance(patch, Rectangle) and patch.get_width() > 0]
    expected: list[float] = []
    for column, _label in PER_CLASS_SPECS:
        for name in names:
            expected.append(100.0 * float(metrics["per_class"][name][column]))
    assert len(bars) == len(expected) == 9
    for bar, value in zip(bars, expected):
        assert bar.get_height() == pytest.approx(value, abs=1e-9)
    plt.close(fig)

"""Tests for baseline metric bar charts against baselines.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from src.config import load_config
from src.plots import (
    CLASSIFIER_ORDER,
    METRIC_SPECS,
    SHORT_NAMES,
    build_baselines_figure,
    plot_baselines,
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

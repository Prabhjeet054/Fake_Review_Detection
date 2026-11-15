"""Tests for src.results_summary using temporary metrics files."""

from __future__ import annotations

import pandas as pd

from src.config import AttrDict
from src.results_summary import build_report, markdown_table


def test_build_report(tmp_config: AttrDict) -> None:
    """Summary report includes experiment rows and a markdown renderer."""
    metrics_dir = tmp_config.paths.metrics_dir
    pd.DataFrame(
        {
            "variant": ["baseline", "mean_pooling"],
            "val_macro_f1": [0.80, 0.85],
            "test_macro_f1": [0.79, 0.84],
        }
    ).to_csv(metrics_dir / "experiments.csv", index=False)
    report = build_report(tmp_config)
    assert report["seed"] == 42
    assert report["best"]["variant"] == "mean_pooling"
    assert report["model_comparison"]
    table = markdown_table(pd.DataFrame(report["model_comparison"]))
    assert table.startswith("|")

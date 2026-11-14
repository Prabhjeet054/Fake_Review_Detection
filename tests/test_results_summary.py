"""Tests for src.results_summary."""

from src.evaluate import save_metrics
from src.results_summary import build_report, markdown_table
import pandas as pd


def test_build_report(cfg: dict) -> None:
    """Summary report includes the comparison table and a markdown renderer."""
    save_metrics({"accuracy": 0.5, "macro_f1": 0.4, "weighted_f1": 0.4, "cohen_kappa": 0.2}, cfg, "baseline_svm_test")
    report = build_report(cfg)
    assert report["project"] == cfg["project"]["name"]
    assert report["model_comparison"]
    table = markdown_table(pd.DataFrame(report["model_comparison"]))
    assert table.startswith("|")

"""Tests for src.results_summary using temporary metrics files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.config import AttrDict, load_config
from src.model import ClassifierHead
from src.results_summary import (
    HEAD_TRAINABLE,
    ConsistencyError,
    build_report,
    check_checkpoint_matches_config,
    check_head_parameter_count,
    check_history_matches_curves,
    check_metrics_match_confusion,
    check_seed42_accuracy_stable,
    expected_head_parameters,
    markdown_table,
    parse_model_summary,
    report_numbers_path,
    run_consistency_checks,
    write_report_numbers,
)
from src.train import save_checkpoint, serialize_config
from src.utils import save_json


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


def _consistent_metrics() -> dict[str, object]:
    """Return a 3x3 metrics payload whose fields match the confusion matrix."""
    return {
        "n_test": 6,
        "n_correct": 6,
        "accuracy": 1.0,
        "macro_precision": 1.0,
        "macro_recall": 1.0,
        "macro_f1": 1.0,
        "weighted_precision": 1.0,
        "weighted_recall": 1.0,
        "weighted_f1": 1.0,
        "cohen_kappa": 1.0,
        "per_class": {
            "Genuine Human": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2},
            "Fake Human-Written": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2},
            "AI-Generated": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2},
        },
        "confusion_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    }


def _write_history(path: Path) -> None:
    """Write a two-epoch history.csv used by the curve check."""
    pd.DataFrame(
        {
            "epoch": [1, 2],
            "train_loss": [0.8, 0.4],
            "train_acc": [0.5, 0.9],
            "val_loss": [0.7, 0.3],
            "val_acc": [0.4, 0.8],
        }
    ).to_csv(path, index=False)


def test_report_numbers_labels_each_source(tmp_config: AttrDict) -> None:
    """report_numbers.md includes every requested table and names its source file."""
    metrics_dir = Path(tmp_config.paths.metrics_dir)
    figures_dir = Path(tmp_config.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "accuracy_curve.png").write_bytes(b"png")
    (metrics_dir / "data_card.md").write_text("# Data card\n\n| Field | Value |\n| --- | --- |\n| Total rows | 30 |\n", encoding="utf-8")
    save_json(
        {
            "train": {"n_rows": 24, "class_counts": {"0": 8, "1": 8, "2": 8}},
            "val": {"n_rows": 3, "class_counts": {"0": 1, "1": 1, "2": 1}},
            "test": {"n_rows": 3, "class_counts": {"0": 1, "1": 1, "2": 1}},
        },
        metrics_dir / "split_summary.json",
    )
    (metrics_dir / "model_summary.txt").write_text(
        "Total params          1,000\nTrainable params      12,355\nNon-trainable params  0\n",
        encoding="utf-8",
    )
    _write_history(metrics_dir / "history.csv")
    save_json(_consistent_metrics(), metrics_dir / "metrics.json")
    save_json(
        {"genuine_vs_deceptive": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0}},
        metrics_dir / "binary_metrics.json",
    )
    pd.DataFrame(
        {"true_class": [0], "predicted_class": [1], "count": [0], "share_of_errors": [0.0]}
    ).to_csv(metrics_dir / "error_breakdown.csv", index=False)
    pd.DataFrame(
        {
            "model": ["SVM", "SVM"],
            "split": ["val", "test"],
            "accuracy": [0.8, 0.9],
            "macro_f1": [0.7, 0.8],
        }
    ).to_csv(metrics_dir / "baselines.csv", index=False)
    pd.DataFrame({"variant": ["baseline"], "val_macro_f1": [0.5], "test_macro_f1": [0.4]}).to_csv(
        metrics_dir / "experiments.csv", index=False
    )

    path = write_report_numbers(tmp_config, checks=["metrics.json vs confusion matrix"])
    text = path.read_text(encoding="utf-8")
    assert path == report_numbers_path(tmp_config)
    assert "Source: `" in text
    assert "data_card.md" in text
    assert "split_summary.json" in text
    assert "Exact test-set size" in text
    assert "config.yaml" in text
    assert "model_summary.txt" in text
    assert "history.csv" in text
    assert "metrics.json" in text
    assert "binary_metrics.json" in text
    assert "error_breakdown.csv" in text
    assert "baselines.csv" in text
    assert "experiments.csv" in text
    assert "accuracy_curve.png" in text
    assert "PASS: metrics.json vs confusion matrix" in text


def test_metrics_json_must_match_confusion_matrix() -> None:
    """A stored accuracy that disagrees with the matrix raises ConsistencyError."""
    check_metrics_match_confusion(_consistent_metrics())
    broken = _consistent_metrics()
    broken["accuracy"] = 0.5
    with pytest.raises(ConsistencyError, match="confusion matrix"):
        check_metrics_match_confusion(broken)


def test_history_last_row_must_match_plotted_curves(
    tmp_config: AttrDict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last history.csv row must equal the last points on the curve figures."""
    path = Path(tmp_config.paths.metrics_dir) / "history.csv"
    _write_history(path)
    history = pd.read_csv(path)
    check_history_matches_curves(history)
    monkeypatch.setattr(
        "src.results_summary.curve_endpoints",
        lambda _history: {
            label: 0.0
            for label in ("Training Accuracy", "Validation Accuracy", "Training Loss", "Validation Loss")
        },
    )
    with pytest.raises(ConsistencyError, match="plotted curves"):
        check_history_matches_curves(history)


def test_checkpoint_config_must_match_yaml(tmp_config: AttrDict) -> None:
    """head_best.pt config must match the model/training slice of config.yaml."""
    head = ClassifierHead(tmp_config)
    path = Path(tmp_config.paths.models_dir) / "head_best.pt"
    save_checkpoint(head, tmp_config, 1, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    check_checkpoint_matches_config(tmp_config, payload)
    payload["config"] = serialize_config(tmp_config)
    payload["config"]["seed"] = 99
    with pytest.raises(ConsistencyError, match="stored config differs"):
        check_checkpoint_matches_config(tmp_config, payload)


def test_head_must_have_expected_parameter_count(tmp_config: AttrDict) -> None:
    """Tiny heads match the architecture formula; a 256-d head must have 197,635 params."""
    actual = expected_head_parameters(tmp_config)
    check_head_parameter_count(tmp_config, actual)
    with pytest.raises(ConsistencyError, match="parameters"):
        check_head_parameter_count(tmp_config, actual + 1)
    real = load_config()
    assert expected_head_parameters(real) == HEAD_TRAINABLE
    with pytest.raises(ConsistencyError, match="197,635"):
        check_head_parameter_count(real, HEAD_TRAINABLE - 1)


def test_seed42_accuracy_mismatch_fails(tmp_config: AttrDict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two seed-42 test accuracies that disagree fail the consistency check."""
    values = iter([0.86, 0.50])
    monkeypatch.setattr(
        "src.results_summary.accuracy_from_checkpoint",
        lambda config: next(values),
    )
    with pytest.raises(ConsistencyError, match="seed 42"):
        check_seed42_accuracy_stable(tmp_config, stored_accuracy=0.86)


def test_run_consistency_checks_pass_on_tiny_artifacts(tmp_config: AttrDict) -> None:
    """A tmp config with a matching checkpoint and metrics passes the loud checks."""
    metrics_dir = Path(tmp_config.paths.metrics_dir)
    save_json(_consistent_metrics(), metrics_dir / "metrics.json")
    _write_history(metrics_dir / "history.csv")
    head = ClassifierHead(tmp_config)
    save_checkpoint(head, tmp_config, 2, Path(tmp_config.paths.models_dir) / "head_best.pt")
    passed = run_consistency_checks(tmp_config)
    assert "metrics.json vs confusion matrix" in passed
    assert "history.csv last row vs plotted curves" in passed
    assert "head_best.pt config vs config.yaml" in passed


def test_parse_model_summary_extracts_counts() -> None:
    """model_summary.txt totals are parsed as integers without commas."""
    text = "Total params          66,560,515\nTrainable params      197,635\nNon-trainable params  66,362,880\n"
    counts = parse_model_summary(text)
    assert counts["trainable"] == HEAD_TRAINABLE
    assert counts["total"] == 66_560_515


@pytest.mark.slow
def test_real_artifacts_pass_consistency_checks() -> None:
    """The saved pipeline artifacts agree with each other under seed 42."""
    passed = run_consistency_checks(load_config())
    assert "metrics.json vs confusion matrix" in passed
    assert "head parameter count (197,635)" in passed

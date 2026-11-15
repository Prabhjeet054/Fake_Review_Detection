"""Tests for run_all.py stage selection, caching, failure, and summary."""

from __future__ import annotations

from pathlib import Path

import pytest

from run_all import (
    STAGES,
    StageResult,
    artifact_ready,
    artifacts_ready,
    collect_summary,
    format_summary,
    parse_stages,
    run_pipeline,
    stage_artifacts,
)
from src.config import AttrDict


def _config(root: Path) -> AttrDict:
    """Build a minimal AttrDict whose paths live under a temp directory."""
    paths = AttrDict(
        {
            "interim_dir": root / "interim",
            "processed_dir": root / "processed",
            "splits_dir": root / "splits",
            "figures_dir": root / "figures",
            "metrics_dir": root / "metrics",
            "models_dir": root / "models",
            "embeddings_dir": root / "embeddings",
        }
    )
    return AttrDict({"paths": paths, "seed": 42})


def _touch(path: Path) -> None:
    """Create an empty file, including parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_default_stage_order() -> None:
    """The default pipeline is data through experiments."""
    assert parse_stages(None) == list(STAGES)
    assert STAGES == (
        "data",
        "eda",
        "baselines",
        "embed",
        "train",
        "evaluate",
        "plots",
        "analysis",
        "explain",
        "experiments",
    )


def test_comma_separated_stages_and_unknown() -> None:
    """--stages accepts a comma-separated subset and rejects unknown names."""
    assert parse_stages("data, eda,train") == ["data", "eda", "train"]
    with pytest.raises(ValueError, match="Unknown"):
        parse_stages("data,nope")


def test_skip_when_all_outputs_exist(tmp_path: Path) -> None:
    """A stage is skipped when every artifact already exists."""
    cfg = _config(tmp_path)
    called: list[str] = []
    for path in stage_artifacts(cfg)["eda"]:
        _touch(path)
    results = run_pipeline(
        ["eda"],
        cfg,
        force=False,
        runners={"eda": lambda: called.append("eda")},
    )
    assert called == []
    assert results[0].status == "skipped"


def test_force_reruns_cached_stage(tmp_path: Path) -> None:
    """--force re-runs a stage even when its outputs already exist."""
    cfg = _config(tmp_path)
    called: list[str] = []
    for path in stage_artifacts(cfg)["eda"]:
        _touch(path)
    results = run_pipeline(
        ["eda"],
        cfg,
        force=True,
        runners={"eda": lambda: called.append("eda")},
    )
    assert called == ["eda"]
    assert results[0].status == "ran"


def test_stop_at_first_failing_stage(tmp_path: Path) -> None:
    """The pipeline stops at the first failing stage and does not run later ones."""
    cfg = _config(tmp_path)
    called: list[str] = []

    def fail() -> None:
        called.append("data")
        raise RuntimeError("boom")

    results = run_pipeline(
        ["data", "eda"],
        cfg,
        force=True,
        runners={"data": fail, "eda": lambda: called.append("eda")},
    )
    assert called == ["data"]
    assert [row.status for row in results] == ["failed"]
    assert results[0].error == "boom"


def test_explain_directory_needs_html(tmp_path: Path) -> None:
    """The explain stage is ready only when explanation HTML files exist."""
    folder = tmp_path / "explanations"
    folder.mkdir()
    assert artifact_ready(folder) is False
    (folder / "id_0_true_0_pred_0.html").write_text("<html></html>", encoding="utf-8")
    assert artifact_ready(folder) is True
    assert artifacts_ready([folder]) is True


def test_summary_reads_sizes_baseline_and_distilbert(tmp_path: Path) -> None:
    """The end summary reports data sizes, best baseline F1, and DistilBERT metrics."""
    cfg = _config(tmp_path)
    metrics = Path(cfg.paths.metrics_dir)
    metrics.mkdir(parents=True)
    (metrics / "split_summary.json").write_text(
        '{"train":{"n_rows":10},"val":{"n_rows":4},"test":{"n_rows":6}}',
        encoding="utf-8",
    )
    (metrics / "baselines.csv").write_text(
        "model,split,macro_f1\nKNN,test,0.70\nSVM,test,0.90\nSVM,val,0.99\n",
        encoding="utf-8",
    )
    (metrics / "metrics.json").write_text(
        '{"accuracy":0.8625,"macro_f1":0.8525,"cohen_kappa":0.7739}',
        encoding="utf-8",
    )
    payload = collect_summary(cfg)
    assert payload["sizes"] == {"train": 10, "val": 4, "test": 6}
    assert payload["best_baseline"] == "SVM"
    assert payload["best_baseline_macro_f1"] == pytest.approx(0.90)
    assert payload["distilbert_accuracy"] == pytest.approx(0.8625)
    text = format_summary(cfg, [StageResult("eda", "skipped", 0.0)])
    assert "Best baseline (test macro F1): SVM 0.9000" in text
    assert "accuracy=0.8625" in text
    assert "macro_F1=0.8525" in text
    assert "kappa=0.7739" in text

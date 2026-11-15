"""Run the fake-review pipeline in order, skipping stages whose outputs exist."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AttrDict, load_config
from src.utils import ensure_dirs, set_seed

STAGES: tuple[str, ...] = (
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

DATA_MODULES: tuple[str, ...] = ("load_data", "clean", "dedup", "split", "check_data")


@dataclass
class StageResult:
    """Outcome of one pipeline stage."""

    name: str
    status: str
    seconds: float
    error: str | None = None


def parse_stages(raw: str | None) -> list[str]:
    """Parse a comma-separated subset, or return the full stage list."""
    if raw is None or not str(raw).strip():
        return list(STAGES)
    stages = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not stages:
        raise ValueError("--stages was empty. Pass a comma-separated subset of the pipeline.")
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s): {unknown}. Valid: {list(STAGES)}")
    return stages


def stage_artifacts(config: AttrDict) -> dict[str, list[Path]]:
    """Return the output paths that mark each stage as complete."""
    paths = config.paths
    splits = Path(paths.splits_dir)
    metrics = Path(paths.metrics_dir)
    figures = Path(paths.figures_dir)
    models = Path(paths.models_dir)
    interim = Path(paths.interim_dir)
    processed = Path(paths.processed_dir)
    embeddings = Path(paths.embeddings_dir)
    embed_files = [
        embeddings / f"{split}_{suffix}.npy"
        for split in ("train", "val", "test")
        for suffix in ("X", "y", "ids")
    ]
    return {
        "data": [
            interim / "merged.csv",
            interim / "cleaned.csv",
            processed / "dataset.csv",
            splits / "train.csv",
            splits / "val.csv",
            splits / "test.csv",
            metrics / "split_summary.json",
            metrics / "data_card.md",
        ],
        "eda": [
            figures / "class_distribution.png",
            figures / "length_distribution.png",
            metrics / "eda_summary.json",
        ],
        "baselines": [
            metrics / "baselines.csv",
            metrics / "baseline_predictions_val.csv",
            metrics / "baseline_predictions_test.csv",
        ],
        "embed": embed_files,
        "train": [
            models / "head_best.pt",
            models / "head_last.pt",
            metrics / "history.csv",
        ],
        "evaluate": [
            metrics / "metrics.json",
            metrics / "test_predictions.csv",
        ],
        "plots": [
            figures / "baselines_val.png",
            figures / "baselines_test.png",
            figures / "accuracy_curve.png",
            figures / "loss_curve.png",
            figures / "confusion_matrix.png",
            figures / "confusion_matrix_normalized.png",
            figures / "per_class_metrics.png",
        ],
        "analysis": [
            metrics / "binary_metrics.json",
            metrics / "error_breakdown.csv",
            metrics / "misclassified.csv",
            metrics / "error_analysis.md",
        ],
        "explain": [figures / "explanations"],
        "experiments": [
            metrics / "experiments.csv",
            metrics / "experiments.md",
        ],
    }


def artifact_ready(path: Path) -> bool:
    """Return True if a file exists, or a directory contains at least one HTML file."""
    if path.suffix:
        return path.is_file()
    return path.is_dir() and any(path.glob("*.html"))


def artifacts_ready(paths: Sequence[Path]) -> bool:
    """Return True when every listed output already exists."""
    return bool(paths) and all(artifact_ready(path) for path in paths)


def _call_module(module_name: str, argv: list[str] | None = None) -> None:
    """Import a src module and invoke its main()."""
    module = importlib.import_module(f"src.{module_name}")
    params = inspect.signature(module.main).parameters
    # Always pass a list so argparse does not read run_all.py's sys.argv.
    result = module.main([] if argv is None else argv) if params else module.main()
    if isinstance(result, int) and result != 0:
        raise RuntimeError(f"{module_name} exited with code {result}")


def run_data_stage() -> None:
    """Build, clean, dedup, split, and check the dataset."""
    for module_name in DATA_MODULES:
        _call_module(module_name)


def default_runners(force: bool) -> dict[str, Callable[[], None]]:
    """Map stage names to callables that invoke the matching src module."""
    return {
        "data": run_data_stage,
        "eda": lambda: _call_module("eda"),
        "baselines": lambda: _call_module("baselines"),
        "embed": lambda: _call_module("embed", ["--force"] if force else []),
        "train": lambda: _call_module("train"),
        "evaluate": lambda: _call_module("evaluate"),
        "plots": lambda: _call_module("plots"),
        "analysis": lambda: _call_module("analysis"),
        "explain": lambda: _call_module("explain"),
        "experiments": lambda: _call_module(
            "experiments",
            ["--force-mean-cache"] if force else [],
        ),
    }


def run_pipeline(
    stages: Sequence[str],
    config: AttrDict,
    force: bool = False,
    runners: dict[str, Callable[[], None]] | None = None,
) -> list[StageResult]:
    """Run selected stages in order, skipping cached outputs unless force is set."""
    selected = list(stages)
    unknown = [stage for stage in selected if stage not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s): {unknown}. Valid: {list(STAGES)}")
    callables = runners if runners is not None else default_runners(force)
    artifacts = stage_artifacts(config)
    results: list[StageResult] = []
    for stage in selected:
        print(f"=== {stage} ===")
        if not force and artifacts_ready(artifacts[stage]):
            print(f"skipping {stage} (outputs already exist)")
            results.append(StageResult(stage, "skipped", 0.0))
            continue
        started = time.perf_counter()
        try:
            callables[stage]()
        except SystemExit as exc:
            elapsed = time.perf_counter() - started
            code = exc.code if isinstance(exc.code, int) else 1
            if code in (0, None):
                print(f"[{stage}] done in {elapsed:.2f}s")
                results.append(StageResult(stage, "ran", elapsed))
                continue
            message = f"exited with code {code}"
            print(f"Stage {stage} failed: {message}", file=sys.stderr)
            results.append(StageResult(stage, "failed", elapsed, message))
            return results
        except Exception as exc:
            elapsed = time.perf_counter() - started
            message = str(exc) or exc.__class__.__name__
            print(f"Stage {stage} failed: {message}", file=sys.stderr)
            results.append(StageResult(stage, "failed", elapsed, message))
            return results
        elapsed = time.perf_counter() - started
        print(f"[{stage}] done in {elapsed:.2f}s")
        results.append(StageResult(stage, "ran", elapsed))
    return results


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON object if the file exists."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _fmt_metric(value: Any) -> str:
    """Format a numeric metric or return n/a."""
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def collect_summary(config: AttrDict) -> dict[str, Any]:
    """Read data sizes, best baseline, and DistilBERT test metrics from disk."""
    metrics_dir = Path(config.paths.metrics_dir)
    splits_dir = Path(config.paths.splits_dir)
    sizes: dict[str, int | None] = {"train": None, "val": None, "test": None}
    split_summary = _load_json(metrics_dir / "split_summary.json")
    if split_summary:
        for split in sizes:
            block = split_summary.get(split) or {}
            if "n_rows" in block:
                sizes[split] = int(block["n_rows"])
    for split, count in list(sizes.items()):
        if count is not None:
            continue
        csv_path = splits_dir / f"{split}.csv"
        if csv_path.is_file():
            sizes[split] = int(sum(1 for _ in csv_path.open(encoding="utf-8")) - 1)

    best_name: str | None = None
    best_f1: float | None = None
    baselines_path = metrics_dir / "baselines.csv"
    if baselines_path.is_file():
        frame = pd.read_csv(baselines_path)
        test = frame.loc[frame["split"] == "test"] if "split" in frame.columns else frame
        if not test.empty and "macro_f1" in test.columns:
            winner = test.loc[test["macro_f1"].idxmax()]
            best_name = str(winner.get("model", winner.get("model_key", "unknown")))
            best_f1 = float(winner["macro_f1"])

    distilbert = _load_json(metrics_dir / "metrics.json") or {}
    return {
        "sizes": sizes,
        "best_baseline": best_name,
        "best_baseline_macro_f1": best_f1,
        "distilbert_accuracy": distilbert.get("accuracy"),
        "distilbert_macro_f1": distilbert.get("macro_f1"),
        "distilbert_kappa": distilbert.get("cohen_kappa"),
    }


def format_summary(config: AttrDict, results: Sequence[StageResult]) -> str:
    """Build the end-of-run summary text."""
    payload = collect_summary(config)
    sizes = payload["sizes"]
    lines = [
        "=== Pipeline summary ===",
        *(f"{row.name}: {row.status} ({row.seconds:.2f}s)" for row in results),
        "",
        (
            f"Data sizes: train={sizes['train'] if sizes['train'] is not None else 'n/a'}  "
            f"val={sizes['val'] if sizes['val'] is not None else 'n/a'}  "
            f"test={sizes['test'] if sizes['test'] is not None else 'n/a'}"
        ),
        (
            f"Best baseline (test macro F1): {payload['best_baseline']} "
            f"{_fmt_metric(payload['best_baseline_macro_f1'])}"
            if payload["best_baseline"] is not None
            else "Best baseline (test macro F1): n/a"
        ),
        (
            "DistilBERT test: "
            f"accuracy={_fmt_metric(payload['distilbert_accuracy'])}  "
            f"macro_F1={_fmt_metric(payload['distilbert_macro_f1'])}  "
            f"kappa={_fmt_metric(payload['distilbert_kappa'])}"
        ),
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse pipeline CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the fake-review-detector pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated subset of stages to run "
        "(data,eda,baselines,embed,train,evaluate,plots,analysis,explain,experiments).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stages even when their outputs already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python run_all.py`."""
    args = parse_args(argv)
    try:
        stages = parse_stages(args.stages)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    config = load_config(args.config)
    set_seed(int(config.seed))
    ensure_dirs(config)
    results = run_pipeline(stages, config, force=bool(args.force))
    print(format_summary(config, results))
    if any(row.status == "failed" for row in results):
        failed = next(row for row in results if row.status == "failed")
        print(f"Pipeline stopped at stage {failed.name}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

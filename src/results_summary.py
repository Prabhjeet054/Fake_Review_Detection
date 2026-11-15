"""Compile report numbers from saved artifacts and fail on inconsistencies."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import AttrDict, find_project_root, load_config
from src.embed import embedding_paths
from src.evaluate import (
    compute_metrics,
    load_best_head,
    load_test_embeddings,
    predict_proba_and_labels,
)
from src.experiments import best_model, collect_results
from src.model import CLS_DIM, ClassifierHead, count_parameters
from src.plots import build_history_curve
from src.train import serialize_config
from src.utils import load_json, save_json, set_seed

HEAD_TRAINABLE = 197_635
HISTORY_EPOCHS = (1, 10, 20, 30, 50)
CONFIG_COMPARE_KEYS = (
    "seed",
    "labels",
    "split",
    "dedup",
    "tfidf",
    "model",
    "training",
    "experiments",
    "data",
)
CURVE_COLUMNS = {
    "Training Accuracy": "train_acc",
    "Validation Accuracy": "val_acc",
    "Training Loss": "train_loss",
    "Validation Loss": "val_loss",
}


class ConsistencyError(RuntimeError):
    """Raised when report artifacts disagree with each other."""


def markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    if df.empty:
        return "_No results yet. Train models first._"
    headers = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in df.iterrows():
        values: list[str] = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float) and float(value).is_integer() and abs(value) >= 1:
                values.append(str(int(value)))
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    """Format a scalar for the report markdown."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or pd.isna(value))):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _display_path(path: Path | str, config: AttrDict | None = None) -> str:
    """Return a project-relative path when the file lives inside the repo."""
    root = Path(getattr(config, "project_root", find_project_root())) if config is not None else find_project_root()
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _source_line(path: Path | str, config: AttrDict | None = None) -> str:
    """Return a markdown line naming the file a table was read from."""
    return f"*Source: `{_display_path(path, config)}`*"


def _optional_json(path: Path) -> dict[str, Any] | list[Any] | None:
    """Load JSON when the file exists."""
    if not path.is_file():
        return None
    payload = load_json(path)
    return payload


def _optional_csv(path: Path) -> pd.DataFrame | None:
    """Load a CSV when the file exists."""
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _optional_text(path: Path) -> str | None:
    """Read a text file when it exists."""
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def report_numbers_path(config: AttrDict) -> Path:
    """Return `outputs/report_numbers.md` next to the configured metrics directory."""
    return Path(config.paths.metrics_dir).parent / "report_numbers.md"


def expected_head_parameters(config: AttrDict) -> int:
    """Return Linear(768, H)+bias plus Linear(H, C)+bias for this config."""
    hidden = int(config.model.hidden_units)
    n_classes = int(config.model.num_classes)
    return CLS_DIM * hidden + hidden + hidden * n_classes + n_classes


def parse_model_summary(text: str) -> dict[str, int]:
    """Parse total / trainable / frozen parameter counts from model_summary.txt."""
    counts: dict[str, int] = {}
    mapping = {
        "total": r"Total params\s+([0-9,]+)",
        "trainable": r"Trainable params\s+([0-9,]+)",
        "non_trainable": r"Non-trainable params\s+([0-9,]+)",
    }
    for key, pattern in mapping.items():
        match = re.search(pattern, text)
        if match:
            counts[key] = int(match.group(1).replace(",", ""))
    return counts


def _jsonable(value: Any) -> Any:
    """Convert config values into a JSON-comparable nested structure."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def config_slice(payload: dict[str, Any] | AttrDict) -> dict[str, Any]:
    """Keep the config keys that must match between checkpoint and yaml."""
    raw = dict(payload)
    return {key: _jsonable(raw[key]) for key in CONFIG_COMPARE_KEYS if key in raw}


def metrics_from_confusion(metrics: dict[str, Any]) -> dict[str, Any]:
    """Recompute n_test, n_correct, accuracy, support, P/R from the stored matrix."""
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ConsistencyError(
            f"metrics.json confusion_matrix must be square, got shape {matrix.shape}."
        )
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    diag = np.diag(matrix)
    n_test = int(matrix.sum())
    n_correct = int(diag.sum())
    per_class: dict[str, dict[str, float]] = {}
    stored = metrics.get("per_class", {})
    names = list(stored) if isinstance(stored, dict) and stored else [str(i) for i in range(len(diag))]
    if len(names) != len(diag):
        names = [str(i) for i in range(len(diag))]
    for index, name in enumerate(names):
        support = int(row_sums[index])
        precision = float(diag[index] / col_sums[index]) if col_sums[index] else 0.0
        recall = float(diag[index] / support) if support else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "support": support}
    return {
        "n_test": n_test,
        "n_correct": n_correct,
        "accuracy": float(n_correct / n_test) if n_test else 0.0,
        "per_class": per_class,
    }


def curve_endpoints(history: pd.DataFrame) -> dict[str, float]:
    """Return the last y-value of each labelled line on the history curve figures."""
    endpoints: dict[str, float] = {}
    for kind in ("accuracy", "loss"):
        fig, ax = build_history_curve(kind, history)
        try:
            for line in ax.get_lines():
                label = str(line.get_label())
                if label.startswith("_"):
                    continue
                y_values = np.asarray(line.get_ydata(), dtype=np.float64)
                if y_values.size:
                    endpoints[label] = float(y_values[-1])
        finally:
            plt.close(fig)
    return endpoints


def _close(left: float, right: float, atol: float = 1e-9) -> bool:
    """Return True when two floats agree within atol."""
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol))


def check_metrics_match_confusion(metrics: dict[str, Any]) -> None:
    """Fail if metrics.json fields disagree with its own confusion matrix."""
    derived = metrics_from_confusion(metrics)
    problems: list[str] = []
    for key in ("n_test", "n_correct"):
        if int(metrics[key]) != int(derived[key]):
            problems.append(f"{key}: stored {metrics[key]} vs confusion-matrix {derived[key]}")
    if not _close(float(metrics["accuracy"]), float(derived["accuracy"])):
        problems.append(
            f"accuracy: stored {metrics['accuracy']} vs confusion-matrix {derived['accuracy']}"
        )
    stored_classes = metrics.get("per_class", {})
    if isinstance(stored_classes, dict):
        for name, derived_row in derived["per_class"].items():
            if name not in stored_classes:
                continue
            stored_row = stored_classes[name]
            if int(stored_row["support"]) != int(derived_row["support"]):
                problems.append(
                    f"{name} support: stored {stored_row['support']} vs row-sum {derived_row['support']}"
                )
            if not _close(float(stored_row["recall"]), float(derived_row["recall"]), atol=1e-8):
                problems.append(
                    f"{name} recall: stored {stored_row['recall']} vs confusion-matrix {derived_row['recall']}"
                )
            if not _close(float(stored_row["precision"]), float(derived_row["precision"]), atol=1e-8):
                problems.append(
                    f"{name} precision: stored {stored_row['precision']} vs confusion-matrix {derived_row['precision']}"
                )
    if problems:
        joined = "; ".join(problems)
        raise ConsistencyError(f"metrics.json disagrees with its own confusion matrix: {joined}")


def check_history_matches_curves(history: pd.DataFrame) -> None:
    """Fail if history.csv's last row disagrees with the plotted curve endpoints."""
    if history.empty:
        raise ConsistencyError("history.csv is empty, so the plotted curves have no last row.")
    last = history.iloc[-1]
    endpoints = curve_endpoints(history)
    problems: list[str] = []
    for label, column in CURVE_COLUMNS.items():
        if label not in endpoints:
            problems.append(f"plotted curve is missing line {label!r}")
            continue
        if column not in last:
            problems.append(f"history.csv is missing column {column!r}")
            continue
        stored = float(last[column])
        plotted = float(endpoints[label])
        if not _close(stored, plotted, atol=1e-9):
            problems.append(f"{label}: history.csv last row {stored} vs plotted {plotted}")
    if problems:
        joined = "; ".join(problems)
        raise ConsistencyError(
            f"history.csv's last row disagrees with the plotted curves: {joined}"
        )


def check_checkpoint_matches_config(config: AttrDict, checkpoint: dict[str, Any]) -> None:
    """Fail if head_best.pt stored a different config than config.yaml."""
    stored = checkpoint.get("config")
    if not isinstance(stored, dict):
        raise ConsistencyError("head_best.pt is missing a stored config dict.")
    expected = config_slice(serialize_config(config))
    actual = config_slice(stored)
    if json.dumps(expected, sort_keys=True, default=str) != json.dumps(actual, sort_keys=True, default=str):
        raise ConsistencyError(
            "checkpoint stored config differs from config.yaml "
            f"(compared keys: {', '.join(CONFIG_COMPARE_KEYS)})."
        )


def check_head_parameter_count(config: AttrDict, actual: int) -> None:
    """Fail if the classification head does not match the architecture (197,635 for 256-d)."""
    expected = expected_head_parameters(config)
    if actual != expected:
        raise ConsistencyError(
            f"classification head has {actual:,} parameters, expected {expected:,} "
            f"for hidden_units={int(config.model.hidden_units)}."
        )
    if (
        int(config.model.hidden_units) == 256
        and int(config.model.num_classes) == 3
        and actual != HEAD_TRAINABLE
    ):
        raise ConsistencyError(
            f"classification head has {actual:,} parameters, expected {HEAD_TRAINABLE:,}."
        )


def accuracy_from_checkpoint(config: AttrDict) -> float | None:
    """Score head_best.pt on cached test embeddings, or None when files are missing."""
    model_path = Path(config.paths.models_dir) / "head_best.pt"
    paths = embedding_paths(config, "test")
    if not model_path.is_file() or not paths["X"].is_file() or not paths["y"].is_file():
        return None
    device = "cpu"
    set_seed(int(config.seed))
    model = load_best_head(config, device)
    X, y_true, _ids = load_test_embeddings(config)
    _probs, y_pred = predict_proba_and_labels(model, X, device, int(config.training.batch_size))
    metrics = compute_metrics(y_true, y_pred, config)
    return float(metrics["accuracy"])


def check_seed42_accuracy_stable(config: AttrDict, stored_accuracy: float | None) -> bool:
    """Fail if two seed-42 evaluations (or eval vs metrics.json) disagree on test accuracy."""
    first = accuracy_from_checkpoint(config)
    second = accuracy_from_checkpoint(config)
    if first is None or second is None:
        return False
    problems: list[str] = []
    if not _close(first, second, atol=1e-12):
        problems.append(f"run A {first} vs run B {second}")
    if stored_accuracy is not None and not _close(first, float(stored_accuracy), atol=1e-12):
        problems.append(f"re-run {first} vs metrics.json {stored_accuracy}")
    if problems:
        joined = "; ".join(problems)
        raise ConsistencyError(
            f"two runs with seed {int(config.seed)} gave different test accuracy: {joined}"
        )
    return True


def run_consistency_checks(config: AttrDict) -> list[str]:
    """Run every available consistency check and raise on the first group of failures."""
    metrics_dir = Path(config.paths.metrics_dir)
    models_dir = Path(config.paths.models_dir)
    passed: list[str] = []
    failures: list[str] = []

    metrics_path = metrics_dir / "metrics.json"
    metrics_payload = _optional_json(metrics_path)
    if isinstance(metrics_payload, dict) and "confusion_matrix" in metrics_payload:
        try:
            check_metrics_match_confusion(metrics_payload)
            passed.append("metrics.json vs confusion matrix")
        except ConsistencyError as error:
            failures.append(str(error))

    history_path = metrics_dir / "history.csv"
    history = _optional_csv(history_path)
    if history is not None:
        try:
            check_history_matches_curves(history)
            passed.append("history.csv last row vs plotted curves")
        except ConsistencyError as error:
            failures.append(str(error))

    checkpoint_path = models_dir / "head_best.pt"
    checkpoint: dict[str, Any] | None = None
    if checkpoint_path.is_file():
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict):
            checkpoint = loaded
            try:
                check_checkpoint_matches_config(config, checkpoint)
                passed.append("head_best.pt config vs config.yaml")
            except ConsistencyError as error:
                failures.append(str(error))

    try:
        actual = count_parameters(ClassifierHead(config), trainable_only=True)
        if checkpoint is not None and "state_dict" in checkpoint:
            actual = int(sum(tensor.numel() for tensor in checkpoint["state_dict"].values()))
        check_head_parameter_count(config, actual)
        passed.append(f"head parameter count ({actual:,})")
    except ConsistencyError as error:
        failures.append(str(error))

    stored_accuracy = None
    if isinstance(metrics_payload, dict) and "accuracy" in metrics_payload:
        stored_accuracy = float(metrics_payload["accuracy"])
    try:
        if check_seed42_accuracy_stable(config, stored_accuracy):
            passed.append("two seed-42 test accuracies")
    except ConsistencyError as error:
        failures.append(str(error))

    if failures:
        raise ConsistencyError("CONSISTENCY CHECK FAILED:\n- " + "\n- ".join(failures))
    return passed


def _history_rows(history: pd.DataFrame) -> pd.DataFrame:
    """Select epochs 1, 10, 20, 30, 50 and the best (lowest val_loss) row."""
    if history.empty or "epoch" not in history.columns:
        return history
    frame = history.copy()
    frame["epoch"] = frame["epoch"].astype(int)
    selected = frame.loc[frame["epoch"].isin(HISTORY_EPOCHS)].copy()
    best_epoch = int(frame.loc[frame["val_loss"].idxmin(), "epoch"])
    best_row = frame.loc[frame["epoch"] == best_epoch].copy()
    best_row.insert(0, "note", f"best (epoch {best_epoch}, min val_loss)")
    selected = selected.copy()
    epoch_notes = [f"epoch {int(epoch)}" for epoch in selected["epoch"]]
    selected.insert(0, "note", epoch_notes)
    combined = pd.concat([selected, best_row], ignore_index=True)
    return combined


def _model_config_rows(config: AttrDict) -> pd.DataFrame:
    """Flatten seed / model / training settings from config.yaml."""
    rows = [{"setting": "seed", "value": int(config.seed)}]
    for section_name in ("model", "training"):
        section = dict(getattr(config, section_name))
        for key, value in section.items():
            rows.append({"setting": f"{section_name}.{key}", "value": value})
    return pd.DataFrame(rows)


def _split_count_rows(summary: dict[str, Any]) -> pd.DataFrame:
    """Turn split_summary.json into one row per split with class counts."""
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        payload = summary.get(split, {})
        counts = payload.get("class_counts", {})
        row: dict[str, Any] = {"split": split, "n_rows": payload.get("n_rows")}
        for label in sorted(counts, key=lambda item: int(item)):
            row[f"class_{label}"] = counts[label]
        rows.append(row)
    return pd.DataFrame(rows)


def _confusion_frame(matrix: list[list[int]], names: list[str]) -> pd.DataFrame:
    """Render a confusion matrix with class-name rows and columns."""
    frame = pd.DataFrame(matrix, index=names, columns=names)
    frame.index.name = "true"
    return frame.reset_index()


def _figure_paths(config: AttrDict) -> list[str]:
    """List figure files under the configured figures directory."""
    root = Path(config.paths.figures_dir)
    if not root.exists():
        return []
    return [
        _display_path(path, config)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".gitkeep"
    ]


def _section(title: str, source: Path | str, body: str, config: AttrDict | None = None) -> list[str]:
    """Return a markdown section headed by title and source file."""
    return [f"## {title}", "", _source_line(source, config), "", body.strip(), ""]


def build_report_numbers_markdown(config: AttrDict, checks: list[str] | None = None) -> str:
    """Assemble the labelled report-numbers markdown from available artifacts."""
    metrics_dir = Path(config.paths.metrics_dir)
    lines: list[str] = [
        "# Report numbers",
        "",
        "Every number below is labelled with the file it was read from.",
        "",
    ]

    def section(title: str, source: Path | str, body: str) -> list[str]:
        """Build a markdown section with a project-relative source path."""
        return _section(title, source, body, config)

    data_card_path = metrics_dir / "data_card.md"
    data_card = _optional_text(data_card_path)
    if data_card:
        body = data_card.strip()
        if body.startswith("# "):
            body = "\n".join(body.splitlines()[1:]).strip()
        lines.extend(section("Dataset and class description", data_card_path, body))

    split_path = metrics_dir / "split_summary.json"
    split_summary = _optional_json(split_path)
    if isinstance(split_summary, dict):
        test = split_summary.get("test", {})
        counts = test.get("class_counts", {})
        count_bits = ", ".join(f"{key}={counts[key]}" for key in sorted(counts, key=lambda item: int(item)))
        body = "\n".join(
            [
                markdown_table(_split_count_rows(split_summary)),
                "",
                f"- Exact test-set size: **{_fmt(test.get('n_rows'))}**",
                f"- Test class counts: **{count_bits or '—'}**",
            ]
        )
        lines.extend(section("Test-set size and class counts", split_path, body))

    config_path = Path(getattr(config, "config_path", "config.yaml"))
    lines.extend(
        section(
            "Model configuration",
            config_path,
            markdown_table(_model_config_rows(config)),
        )
    )

    summary_path = metrics_dir / "model_summary.txt"
    summary_text = _optional_text(summary_path)
    if summary_text:
        parsed = parse_model_summary(summary_text)
        param_rows = pd.DataFrame(
            [{"count": key, "parameters": value} for key, value in parsed.items()]
        )
        body = markdown_table(param_rows) if not param_rows.empty else summary_text.strip()
        lines.extend(section("Parameter counts", summary_path, body))

    history_path = metrics_dir / "history.csv"
    history = _optional_csv(history_path)
    if history is not None and not history.empty:
        best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
        body = "\n".join(
            [
                f"- Best epoch (lowest val_loss): **{best_epoch}**",
                "",
                markdown_table(_history_rows(history)),
            ]
        )
        lines.extend(section("Training history", history_path, body))

    metrics_path = metrics_dir / "metrics.json"
    metrics = _optional_json(metrics_path)
    if isinstance(metrics, dict):
        overall = pd.DataFrame(
            [
                {
                    "n_test": metrics.get("n_test"),
                    "n_correct": metrics.get("n_correct"),
                    "accuracy": metrics.get("accuracy"),
                    "macro_precision": metrics.get("macro_precision"),
                    "macro_recall": metrics.get("macro_recall"),
                    "macro_f1": metrics.get("macro_f1"),
                    "weighted_precision": metrics.get("weighted_precision"),
                    "weighted_recall": metrics.get("weighted_recall"),
                    "weighted_f1": metrics.get("weighted_f1"),
                    "cohen_kappa": metrics.get("cohen_kappa"),
                }
            ]
        )
        per_class = metrics.get("per_class", {})
        class_rows = []
        if isinstance(per_class, dict):
            for name, row in per_class.items():
                class_rows.append({"class": name, **row})
        names = list(per_class) if isinstance(per_class, dict) else []
        matrix = metrics.get("confusion_matrix", [])
        parts = [
            markdown_table(overall),
            "",
            "### Per-class",
            "",
            markdown_table(pd.DataFrame(class_rows)) if class_rows else "_No per-class rows._",
            "",
            "### Confusion matrix",
            "",
            markdown_table(_confusion_frame(matrix, names)) if matrix else "_No confusion matrix._",
        ]
        lines.extend(section("Test results", metrics_path, "\n".join(parts)))

    binary_path = metrics_dir / "binary_metrics.json"
    binary = _optional_json(binary_path)
    if isinstance(binary, dict):
        binary_rows = [{"task": name, **values} for name, values in binary.items() if isinstance(values, dict)]
        lines.extend(
            section("Grouped binary metrics", binary_path, markdown_table(pd.DataFrame(binary_rows)))
        )

    errors_path = metrics_dir / "error_breakdown.csv"
    errors = _optional_csv(errors_path)
    if errors is not None:
        lines.extend(section("Error breakdown", errors_path, markdown_table(errors)))

    baselines_path = metrics_dir / "baselines.csv"
    baselines = _optional_csv(baselines_path)
    if baselines is not None:
        parts = []
        for split in ("val", "test"):
            subset = baselines.loc[baselines["split"].astype(str).str.lower() == split]
            parts.extend([f"### {split}", "", markdown_table(subset), ""])
        lines.extend(section("Baselines (val and test)", baselines_path, "\n".join(parts).strip()))

    experiments_path = metrics_dir / "experiments.csv"
    experiments = _optional_csv(experiments_path)
    if experiments is not None:
        lines.extend(section("Experiments", experiments_path, markdown_table(experiments)))

    figures_dir = Path(config.paths.figures_dir)
    figure_list = _figure_paths(config)
    if figure_list:
        body = "\n".join(f"- `{path}`" for path in figure_list)
        lines.extend(section("Figure files", figures_dir, body))

    if checks:
        body = "\n".join(f"- PASS: {name}" for name in checks)
        lines.extend(section("Consistency checks", "computed from the files above", body))

    return "\n".join(lines).rstrip() + "\n"


def write_report_numbers(config: AttrDict, checks: list[str] | None = None) -> Path:
    """Write outputs/report_numbers.md and return its path."""
    path = report_numbers_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report_numbers_markdown(config, checks), encoding="utf-8")
    return path


def build_report(cfg: AttrDict) -> dict[str, Any]:
    """Assemble experiment rows and DistilBERT test metrics."""
    results = collect_results(cfg)
    winner = best_model(results, "val_macro_f1")
    metrics_path = Path(cfg.paths.metrics_dir) / "metrics.json"
    distilbert = load_json(metrics_path) if metrics_path.is_file() else {}
    return {
        "seed": int(cfg.seed),
        "best": winner,
        "model_comparison": results,
        "distilbert": distilbert,
    }


def main(argv: list[str] | None = None) -> None:
    """Write the report-number markdown and fail loudly on inconsistent artifacts."""
    parser = argparse.ArgumentParser(description="Write a results summary table.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    report = build_report(cfg)
    metrics_dir = Path(cfg.paths.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(report["model_comparison"])
    csv_path = metrics_dir / "results_summary.csv"
    comparison.to_csv(csv_path, index=False)
    md = ["# Fake Review Detector Results", ""]
    if report["best"]:
        md.append(
            f"Best experiment by **val_macro_f1**: `{report['best']['variant']}` "
            f"({float(report['best']['val_macro_f1']):.4f})."
        )
        md.append("")
    md.append("## Model comparison")
    md.append("")
    md.append(markdown_table(comparison))
    md_path = metrics_dir / "results_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    json_path = save_json(report, metrics_dir / "results_summary.json")
    checks = run_consistency_checks(cfg)
    report_path = write_report_numbers(cfg, checks)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print(f"wrote {report_path}")
    for name in checks:
        print(f"CHECK PASS: {name}")


if __name__ == "__main__":
    main()

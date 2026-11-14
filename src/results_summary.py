"""Compile a markdown/CSV results table from saved metric files."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import load_config, parse_config_args, path_from_cfg
from src.experiments import collect_results, best_model
from src.utils import load_json, save_dataframe, save_json, seed_from_cfg, set_seed, setup_logging


def grouped_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten grouped-binary metrics for the deep model if present."""
    path = path_from_cfg(cfg, "paths.metrics") / "deep_model_test.json"
    if not path.exists():
        return []
    payload = load_json(path)
    grouped = payload.get("grouped_binary", {})
    rows: list[dict[str, Any]] = []
    for name, metrics in grouped.items():
        row = {"grouping": name}
        row.update({k: v for k, v in metrics.items() if k not in {"positive_labels", "negative_labels"}})
        rows.append(row)
    return rows


def markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    if df.empty:
        return "_No results yet. Train models first._"
    headers = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(cfg: dict[str, Any]) -> dict[str, Any]:
    """Assemble the full results summary payload."""
    results = collect_results(cfg)
    metric = cfg["experiments"]["metric"]
    winner = best_model(results, metric)
    return {
        "project": cfg["project"]["name"],
        "seed": cfg["project"]["seed"],
        "metric": metric,
        "best": winner,
        "model_comparison": results,
        "deep_grouped_binary": grouped_rows(cfg),
    }


def main(argv: list[str] | None = None) -> None:
    """Write CSV + markdown summaries under outputs/metrics."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    report = build_report(cfg)
    metrics_dir = path_from_cfg(cfg, "paths.metrics")
    comparison = pd.DataFrame(report["model_comparison"])
    csv_path = save_dataframe(comparison, metrics_dir / "results_summary.csv")
    md = ["# Fake Review Detector Results", ""]
    if report["best"]:
        md.append(
            f"Best model by **{report['metric']}**: `{report['best']['model']}` "
            f"({report['best'][report['metric']]:.4f})."
        )
        md.append("")
    md.append("## Model comparison")
    md.append("")
    md.append(markdown_table(comparison))
    grouped = pd.DataFrame(report["deep_grouped_binary"])
    if not grouped.empty:
        md.extend(["", "## Grouped binary metrics (deep model)", "", markdown_table(grouped)])
    md_path = metrics_dir / "results_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    json_path = save_json(report, metrics_dir / "results_summary.json")
    logger.info("Wrote results summary -> %s", json_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

"""Compile a markdown/CSV results table from saved metric files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AttrDict, load_config
from src.experiments import best_model, collect_results
from src.utils import load_json, save_json, set_seed


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
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


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
    """Write CSV + markdown summaries under outputs/metrics."""
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
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()

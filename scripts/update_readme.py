"""Fill the README results table from metrics.json and baselines.csv."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"


def find_project_root() -> Path:
    """Return the repository root that contains this scripts/ directory."""
    return Path(__file__).resolve().parent.parent


def load_metrics(path: Path) -> dict[str, Any]:
    """Load DistilBERT test metrics from metrics.json."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_baselines(path: Path) -> pd.DataFrame:
    """Load the test-split baseline rows from baselines.csv."""
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str).str.lower() == "test"].copy()
    return frame.reset_index(drop=True)


def _fmt(value: Any) -> str:
    """Format a numeric metric or return an em dash."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{float(value):.4f}"


def format_results_table(metrics: dict[str, Any], baselines: pd.DataFrame) -> str:
    """Build a markdown comparison table for DistilBERT and TF-IDF baselines."""
    lines = [
        "| Model | Test accuracy | Test macro F1 | Cohen's kappa |",
        "| --- | --- | --- | --- |",
        (
            f"| DistilBERT + MLP | {_fmt(metrics.get('accuracy'))} | "
            f"{_fmt(metrics.get('macro_f1'))} | {_fmt(metrics.get('cohen_kappa'))} |"
        ),
    ]
    if baselines.empty:
        return "\n".join(lines)
    ordered = baselines.sort_values("macro_f1", ascending=False)
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['model']} | {_fmt(row.get('accuracy'))} | "
            f"{_fmt(row.get('macro_f1'))} | — |"
        )
    return "\n".join(lines)


def replace_results_section(readme: str, table: str) -> str:
    """Replace the README block between the results markers."""
    start = readme.find(RESULTS_START)
    end = readme.find(RESULTS_END)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("README.md is missing the RESULTS:START / RESULTS:END markers.")
    before = readme[: start + len(RESULTS_START)]
    after = readme[end:]
    return f"{before}\n\n{table}\n\n{after}"


def main() -> None:
    """Rewrite the results table in README.md from the latest metric files."""
    root = find_project_root()
    metrics = load_metrics(root / "outputs" / "metrics" / "metrics.json")
    baselines = load_baselines(root / "outputs" / "metrics" / "baselines.csv")
    table = format_results_table(metrics, baselines)
    readme_path = root / "README.md"
    updated = replace_results_section(readme_path.read_text(encoding="utf-8"), table)
    readme_path.write_text(updated, encoding="utf-8")
    print(f"updated {readme_path}")


if __name__ == "__main__":
    main()

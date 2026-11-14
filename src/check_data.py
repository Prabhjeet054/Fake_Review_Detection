"""Data-quality checks: labels, emptiness, and split leakage."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import load_config, parse_config_args, path_from_cfg
from src.schema import allowed_labels
from src.split import leakage_groups
from src.utils import load_dataframe, save_json, seed_from_cfg, set_seed, setup_logging


def _issue(level: str, code: str, message: str) -> dict[str, str]:
    """Build a structured validation issue."""
    return {"level": level, "code": code, "message": message}


def check_labels(df: pd.DataFrame, cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Flag missing or invalid class labels."""
    issues: list[dict[str, str]] = []
    label_col = cfg["columns"]["label"]
    if df[label_col].isna().any():
        issues.append(_issue("error", "missing_labels", "Some reviews have a missing label."))
    valid = allowed_labels(cfg)
    invalid = sorted(set(df[label_col].dropna().astype(int)) - valid)
    if invalid:
        issues.append(_issue("error", "invalid_labels", f"Invalid labels present: {invalid}"))
    counts = df[label_col].value_counts()
    min_per_class = int(cfg["validation"]["min_per_class"])
    for label in sorted(valid):
        n = int(counts.get(label, 0))
        if n < min_per_class:
            issues.append(_issue("warning", "rare_class", f"Label {label} has {n} examples."))
    return issues


def check_empty_text(df: pd.DataFrame, cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Flag empty cleaned (or raw) review text."""
    text_col = cfg["columns"]["clean_text"] if cfg["columns"]["clean_text"] in df.columns else cfg["columns"]["text"]
    empty = df[text_col].fillna("").astype(str).str.strip().eq("")
    if empty.any():
        return [_issue("error", "empty_text", f"{int(empty.sum())} reviews have empty text.")]
    return []


def check_group_leakage(splits: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Flag group ids that appear in more than one split."""
    if any(cfg["columns"]["group"] not in frame.columns for frame in splits.values()):
        return [_issue("warning", "missing_groups", "group_id is missing; leakage was not checked.")]
    issues: list[dict[str, str]] = []
    for pair, overlap in leakage_groups(splits, cfg).items():
        if overlap:
            issues.append(_issue("error", "group_leakage", f"{pair} share {len(overlap)} groups."))
    return issues


def check_text_leakage(splits: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Flag exact cleaned texts that appear in more than one split."""
    text_col = cfg["columns"]["clean_text"] if cfg["columns"]["clean_text"] in next(iter(splits.values())).columns else cfg["columns"]["text"]
    sets = {name: set(frame[text_col].fillna("").astype(str)) for name, frame in splits.items()}
    issues: list[dict[str, str]] = []
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        overlap = sets[left] & sets[right]
        overlap.discard("")
        if overlap:
            issues.append(
                _issue("warning", "text_leakage", f"{left}/{right} share {len(overlap)} exact cleaned texts.")
            )
    return issues


def run_checks(splits: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Run all configured quality checks across concatenated splits."""
    combined = pd.concat(splits.values(), ignore_index=True)
    issues: list[dict[str, str]] = []
    issues.extend(check_labels(combined, cfg))
    issues.extend(check_empty_text(combined, cfg))
    issues.extend(check_group_leakage(splits, cfg))
    issues.extend(check_text_leakage(splits, cfg))
    return issues


def checks_failed(issues: list[dict[str, str]], cfg: dict[str, Any]) -> bool:
    """Return True if configured fatal issues are present."""
    codes = {issue["code"] for issue in issues if issue["level"] == "error"}
    if cfg["validation"]["fail_on_invalid_labels"] and ("invalid_labels" in codes or "missing_labels" in codes):
        return True
    if cfg["validation"]["fail_on_empty"] and "empty_text" in codes:
        return True
    if cfg["validation"]["fail_on_leakage"] and "group_leakage" in codes:
        return True
    return False


def load_splits(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load the three split files from disk."""
    split_dir = path_from_cfg(cfg, "paths.splits")
    splits = {}
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        splits[name] = load_dataframe(path)
    return splits


def main(argv: list[str] | None = None) -> None:
    """Run quality checks and write a report JSON."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    splits = load_splits(cfg)
    issues = run_checks(splits, cfg)
    report = {
        "n_issues": len(issues),
        "failed": checks_failed(issues, cfg),
        "issues": issues,
        "split_sizes": {name: int(len(frame)) for name, frame in splits.items()},
    }
    out = save_json(report, path_from_cfg(cfg, "paths.metrics") / "data_checks.json")
    logger.info("Data checks wrote %s (%s issues)", out, len(issues))
    print(f"wrote {out} issues={len(issues)} failed={report['failed']}")
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

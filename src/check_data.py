"""Run data sanity checks and write a markdown data card."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import AttrDict, load_config
from src.schema import ID2LABEL, REQUIRED_COLUMNS, validate_dataframe
from src.utils import ensure_dirs, set_seed

SPLIT_NAMES = ("train", "val", "test")
CHECK_COLUMNS = list(REQUIRED_COLUMNS) + ["text_clean", "group_id"]


def load_splits(config: AttrDict) -> dict[str, pd.DataFrame]:
    """Load train/val/test CSVs from config.paths.splits_dir."""
    split_dir = Path(config.paths.splits_dir)
    splits: dict[str, pd.DataFrame] = {}
    for name in SPLIT_NAMES:
        path = split_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}. Run python -m src.split first.")
        splits[name] = pd.read_csv(path)
    return splits


def check_schema(splits: dict[str, pd.DataFrame]) -> list[str]:
    """Validate each split and the concatenated frame against the canonical schema."""
    issues: list[str] = []
    for name, frame in splits.items():
        try:
            validate_dataframe(frame)
        except ValueError as exc:
            issues.append(f"schema ({name}): {exc}")
    combined = pd.concat(splits.values(), ignore_index=True)
    try:
        validate_dataframe(combined)
    except ValueError as exc:
        issues.append(f"schema (combined): {exc}")
    return issues


def check_missing_values(splits: dict[str, pd.DataFrame]) -> list[str]:
    """Flag NaN or blank values in required columns."""
    issues: list[str] = []
    for name, frame in splits.items():
        for column in CHECK_COLUMNS:
            if column not in frame.columns:
                issues.append(f"missing column '{column}' in {name}")
                continue
            n_nan = int(frame[column].isna().sum())
            if n_nan:
                issues.append(f"{name}: {n_nan} missing values in '{column}'")
            if column in {"text", "text_clean"}:
                empty = frame[column].fillna("").astype(str).str.strip().eq("")
                n_empty = int(empty.sum())
                if n_empty:
                    issues.append(f"{name}: {n_empty} empty values in '{column}'")
    return issues


def check_leakage(splits: dict[str, pd.DataFrame]) -> list[str]:
    """Flag group_id, id, or text_clean values that appear in more than one split."""
    issues: list[str] = []
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for column in ("group_id", "id", "text_clean"):
        sets = {name: set(frame[column].astype(str)) for name, frame in splits.items()}
        for left, right in pairs:
            overlap = sets[left] & sets[right]
            if overlap:
                issues.append(
                    f"leakage: {len(overlap)} shared {column} value(s) between {left} and {right}"
                )
    return issues


def check_class_counts(splits: dict[str, pd.DataFrame], config: AttrDict) -> list[str]:
    """Require every configured class to appear in every split."""
    issues: list[str] = []
    expected = sorted(int(label) for label in config.labels)
    for name, frame in splits.items():
        counts = frame["label"].value_counts()
        for label in expected:
            n = int(counts.get(label, 0))
            if n <= 0:
                issues.append(f"{name}: class {label} has {n} rows")
        unexpected = sorted({int(value) for value in frame["label"].dropna()} - set(expected))
        if unexpected:
            issues.append(f"{name}: unexpected labels {unexpected}")
    dataset_path = Path(config.paths.processed_dir) / "dataset.csv"
    if dataset_path.is_file():
        dataset = pd.read_csv(dataset_path)
        n_splits = sum(len(frame) for frame in splits.values())
        if n_splits != len(dataset):
            issues.append(
                f"row count: train+val+test={n_splits} but dataset.csv has {len(dataset)} rows"
            )
    return issues


def run_checks(splits: dict[str, pd.DataFrame], config: AttrDict) -> list[str]:
    """Run schema, leakage, class-count, and missing-value checks."""
    issues: list[str] = []
    issues.extend(check_schema(splits))
    issues.extend(check_missing_values(splits))
    issues.extend(check_leakage(splits))
    issues.extend(check_class_counts(splits, config))
    return issues


def _class_counts_line(frame: pd.DataFrame) -> str:
    """Format label counts as '0=n, 1=n, 2=n'."""
    counts = frame["label"].value_counts().sort_index()
    return ", ".join(f"{int(label)}={int(count)}" for label, count in counts.items())


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavored markdown table."""
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def build_data_card(splits: dict[str, pd.DataFrame], config: AttrDict) -> str:
    """Return markdown for the dataset and class description tables."""
    combined = pd.concat(splits.values(), ignore_index=True)
    test = splits["test"]
    n_test = max(len(test), 1)
    sources = (
        "Salminen Fake Reviews Dataset "
        f"(`{Path(config.data.salminen_file).name}`); "
        "Ott Deceptive Opinion Spam Corpus "
        f"(`{Path(config.data.ott_file).name}`)"
    )
    ratios = (
        f"train {float(config.split.train):.2f} / "
        f"val {float(config.split.val):.2f} / "
        f"test {float(config.split.test):.2f}"
    )
    rows_per_split = "; ".join(f"{name} {len(splits[name])}" for name in SPLIT_NAMES)
    class_counts_per_split = "; ".join(
        f"{name}: {_class_counts_line(splits[name])}" for name in SPLIT_NAMES
    )
    test_pct = "; ".join(
        f"{int(label)} {ID2LABEL.get(int(label), str(label))} "
        f"{100.0 * int((test['label'] == label).sum()) / n_test:.1f}%"
        for label in sorted(int(key) for key in config.labels)
    )
    dataset_rows = [
        ["Sources", sources],
        ["Input", "`data/processed/dataset.csv`"],
        ["Number of classes", str(int(config.model.num_classes))],
        ["Split ratios", ratios],
        ["Total rows", str(len(combined))],
        ["Rows per split", rows_per_split],
        ["Class counts per split", class_counts_per_split],
        ["Class percentages in the test split", test_pct],
    ]
    class_rows: list[list[str]] = []
    for label in sorted(int(key) for key in config.labels):
        subset = test.loc[test["label"] == label]
        source = ", ".join(sorted(subset["source"].astype(str).unique())) if len(subset) else ""
        domain = ", ".join(sorted(subset["domain"].astype(str).unique())) if len(subset) else ""
        class_rows.append(
            [
                str(label),
                ID2LABEL.get(label, str(label)),
                source,
                domain,
                str(len(subset)),
            ]
        )
    return "\n".join(
        [
            "# Data card",
            "",
            "## Dataset description",
            "",
            _md_table(["Field", "Value"], dataset_rows),
            "",
            "## Class description",
            "",
            _md_table(
                ["Label", "Class name", "Source", "Domain", "Count in test"],
                class_rows,
            ),
            "",
        ]
    )


def write_data_card(splits: dict[str, pd.DataFrame], config: AttrDict) -> Path:
    """Write outputs/metrics/data_card.md and return its path."""
    ensure_dirs(config)
    path = Path(config.paths.metrics_dir) / "data_card.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_data_card(splits, config), encoding="utf-8")
    return path


def run_check_data(config: AttrDict | None = None) -> tuple[list[str], Path]:
    """Load splits, write the data card, and return any failed checks."""
    cfg = config if config is not None else load_config()
    set_seed(int(cfg.seed))
    splits = load_splits(cfg)
    issues = run_checks(splits, cfg)
    path = write_data_card(splits, cfg)
    return issues, path


def main() -> int:
    """Run sanity checks, write the data card, and exit 1 if any check fails."""
    issues, path = run_check_data()
    print(f"wrote {path}")
    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        print(f"{len(issues)} check(s) failed.")
        return 1
    print("All data checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Exploratory figures and summary statistics for the cleaned reviews."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import AttrDict, load_config
from src.schema import ID2LABEL
from src.utils import ensure_dirs, save_json, set_seed

STOP_WORDS = {word.lower() for word in ENGLISH_STOP_WORDS}


def _class_name(label: int) -> str:
    """Return the configured display name for a label id."""
    return ID2LABEL.get(int(label), str(label))


def _word_count(text: str) -> int:
    """Return the number of whitespace-separated tokens."""
    stripped = str(text).strip()
    if not stripped:
        return 0
    return len(stripped.split())


def _freq_tokens(text: str) -> list[str]:
    """Return alphabetic tokens with English stop words removed."""
    tokens: list[str] = []
    for raw in str(text).split():
        token = raw.strip(".,!?'").lower()
        if token and token.isalpha() and token not in STOP_WORDS:
            tokens.append(token)
    return tokens


def _crosstab_dict(df: pd.DataFrame, rows: str, cols: str) -> dict[str, dict[str, int]]:
    """Return a JSON-friendly nested count table."""
    table = pd.crosstab(df[rows], df[cols])
    return {
        str(row): {str(col): int(table.loc[row, col]) for col in table.columns}
        for row in table.index
    }


def top_words_per_class(df: pd.DataFrame, n: int = 20) -> dict[str, list[list[str | int]]]:
    """Return the n most frequent non-stop words per class."""
    result: dict[str, list[list[str | int]]] = {}
    for label, group in df.groupby("label"):
        counts = Counter()
        for text in group["text_clean"].fillna(""):
            counts.update(_freq_tokens(text))
        result[str(int(label))] = [[word, int(count)] for word, count in counts.most_common(n)]
    return result


def build_eda_summary(df: pd.DataFrame) -> dict:
    """Compute class counts, length stats, crosstabs, and top words."""
    work = df.copy()
    work["word_count"] = work["text_clean"].fillna("").map(_word_count)
    counts = work["label"].value_counts().sort_index()
    word_stats: dict[str, dict[str, float | int]] = {}
    for label, group in work.groupby("label"):
        lengths = group["word_count"]
        word_stats[str(int(label))] = {
            "name": _class_name(int(label)),
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "median": float(lengths.median()) if len(lengths) else 0.0,
            "max": int(lengths.max()) if len(lengths) else 0,
        }
    return {
        "n_rows": len(work),
        "counts_per_class": {
            str(int(label)): {"name": _class_name(int(label)), "count": int(count)}
            for label, count in counts.items()
        },
        "word_count_per_class": word_stats,
        "label_x_domain": _crosstab_dict(work, "label", "domain"),
        "label_x_source": _crosstab_dict(work, "label", "source"),
        "top_words_per_class": top_words_per_class(work),
    }


def plot_class_distribution(df: pd.DataFrame, out_path: Path) -> Path:
    """Write a bar chart of class counts with names on the x-axis and counts on the bars."""
    counts = df["label"].value_counts().sort_index()
    names = [_class_name(int(label)) for label in counts.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, counts.to_numpy())
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class distribution")
    for bar, count in zip(bars, counts.to_numpy()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(count)),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_length_distribution(df: pd.DataFrame, out_path: Path) -> Path:
    """Write a box plot of word count per class."""
    work = df.copy()
    work["word_count"] = work["text_clean"].fillna("").map(_word_count)
    work["class_name"] = work["label"].map(lambda value: _class_name(int(value)))
    order = [_class_name(int(label)) for label in sorted(work["label"].unique())]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=work, x="class_name", y="word_count", order=order, ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("Word count")
    ax.set_title("Review length by class")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def warn_single_domain_classes(df: pd.DataFrame) -> list[int]:
    """Print a WARNING for each class whose rows all come from one domain."""
    flagged: list[int] = []
    for label, group in df.groupby("label"):
        domains = sorted({str(value) for value in group["domain"].dropna().unique()})
        if len(domains) == 1:
            label_id = int(label)
            flagged.append(label_id)
            print(
                f"WARNING: class {label_id} ({_class_name(label_id)}) has rows from a single "
                f"domain ({domains[0]}). This is a shortcut risk: the model could learn the "
                f"domain instead of the class."
            )
    return flagged


def run_eda(df: pd.DataFrame, config: AttrDict | None = None) -> dict:
    """Write EDA figures, metrics JSON, and domain-shortcut warnings."""
    cfg = config if config is not None else load_config()
    ensure_dirs(cfg)
    figures_dir = Path(cfg.paths.figures_dir)
    metrics_dir = Path(cfg.paths.metrics_dir)
    plot_class_distribution(df, figures_dir / "class_distribution.png")
    plot_length_distribution(df, figures_dir / "length_distribution.png")
    summary = build_eda_summary(df)
    save_json(summary, metrics_dir / "eda_summary.json")
    warn_single_domain_classes(df)
    print(f"wrote {figures_dir / 'class_distribution.png'}")
    print(f"wrote {figures_dir / 'length_distribution.png'}")
    print(f"wrote {metrics_dir / 'eda_summary.json'}")
    return summary


def main() -> None:
    """Read cleaned.csv and write EDA figures and metrics."""
    cfg = load_config()
    set_seed(int(cfg.seed))
    path = Path(cfg.paths.interim_dir) / "cleaned.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {path}. Run python -m src.clean first."
        )
    run_eda(pd.read_csv(path), cfg)


if __name__ == "__main__":
    main()

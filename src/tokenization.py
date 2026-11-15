"""DistilBERT tokenization helpers. Named tokenization.py so it does not shadow stdlib tokenize."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.config import AttrDict, load_config
from src.schema import ID2LABEL
from src.utils import ensure_dirs, save_json, set_seed

SPLIT_NAMES = ("train", "val", "test")


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str) -> Any:
    """Load and cache a Hugging Face tokenizer by model name."""
    return AutoTokenizer.from_pretrained(model_name)


def get_tokenizer(config: AttrDict) -> Any:
    """Return AutoTokenizer.from_pretrained(config.model.name), loaded once and cached."""
    return _load_tokenizer(str(config.model.name))


def tokenize_texts(texts: list[str], config: AttrDict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded, truncated input_ids and attention_mask tensors."""
    tokenizer = get_tokenizer(config)
    encoded = tokenizer(
        [str(text) for text in texts],
        padding="max_length",
        truncation=True,
        max_length=int(config.model.max_length),
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def _untruncated_lengths(texts: list[str], tokenizer: Any) -> list[int]:
    """Return token counts including special tokens, with no truncation."""
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]


def _load_split(config: AttrDict, name: str) -> pd.DataFrame:
    """Load one split CSV."""
    path = Path(config.paths.splits_dir) / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}. Run python -m src.split first.")
    return pd.read_csv(path)


def token_length_report(config: AttrDict | None = None) -> dict[str, Any]:
    """Summarize untruncated token lengths per split and class, then save figure and JSON."""
    cfg = config if config is not None else load_config()
    set_seed(int(cfg.seed))
    ensure_dirs(cfg)
    tokenizer = get_tokenizer(cfg)
    max_length = int(cfg.model.max_length)
    report: dict[str, Any] = {"model": str(cfg.model.name), "max_length": max_length, "splits": {}}
    plot_rows: list[dict] = []
    for split in SPLIT_NAMES:
        frame = _load_split(cfg, split)
        texts = frame["text_clean"].fillna("").astype(str).tolist()
        lengths = _untruncated_lengths(texts, tokenizer)
        work = frame.copy()
        work["n_tokens"] = lengths
        split_stats: dict[str, dict] = {}
        for label, group in work.groupby("label"):
            n = len(group)
            over = int((group["n_tokens"] > max_length).sum())
            split_stats[str(int(label))] = {
                "name": ID2LABEL.get(int(label), str(label)),
                "n": n,
                "mean_tokens": float(group["n_tokens"].mean()) if n else 0.0,
                "median_tokens": float(group["n_tokens"].median()) if n else 0.0,
                "max_tokens": int(group["n_tokens"].max()) if n else 0,
                "n_over_max_length": over,
                "share_over_max_length": float(over / n) if n else 0.0,
            }
        report["splits"][split] = split_stats
        for _, row in work.iterrows():
            plot_rows.append(
                {
                    "split": split,
                    "label": int(row["label"]),
                    "n_tokens": int(row["n_tokens"]),
                }
            )
    json_path = Path(cfg.paths.metrics_dir) / "token_lengths.json"
    save_json(report, json_path)
    fig_path = Path(cfg.paths.figures_dir) / "token_lengths.png"
    _plot_token_lengths(pd.DataFrame(plot_rows), max_length, fig_path)
    print(f"wrote {json_path}")
    print(f"wrote {fig_path}")
    return report


def _plot_token_lengths(df: pd.DataFrame, max_length: int, out_path: Path) -> Path:
    """Write a histogram of untruncated token counts per class with a max_length line."""
    labels = sorted(int(value) for value in df["label"].unique())
    fig, axes = plt.subplots(1, max(len(labels), 1), figsize=(12, 4), sharey=True)
    if len(labels) <= 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        subset = df.loc[df["label"] == label, "n_tokens"]
        ax.hist(subset.to_numpy(), bins=30, color="#4C78A8", edgecolor="white")
        ax.axvline(max_length, color="#E45756", linestyle="--", linewidth=1.5, label=f"max_length={max_length}")
        ax.set_title(ID2LABEL.get(label, str(label)))
        ax.set_xlabel("Untruncated tokens")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Reviews")
    fig.suptitle("Token length by class")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    """Print a sample batch shape and write the token-length report."""
    cfg = load_config()
    ids, mask = tokenize_texts(["the hotel was clean and the staff were helpful"], cfg)
    print(f"input_ids.shape={tuple(ids.shape)} attention_mask.shape={tuple(mask.shape)}")
    token_length_report(cfg)


if __name__ == "__main__":
    main()

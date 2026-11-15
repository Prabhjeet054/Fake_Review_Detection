"""Shared pytest fixtures: a 30-row corpus and a temporary config under tmp_path."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from src.config import AttrDict, load_config
from src.utils import ensure_dirs

N_PER_CLASS = 10
N_CLASSES = 3
TINY_N_ROWS = N_PER_CLASS * N_CLASSES

CLASS_TEXT = {
    0: "I bought this blender last month and it still works well after daily use.",
    1: "Wow I cannot believe how perfect this hotel stay was everyone must book now.",
    2: "This product is amazing best product ever highly recommend this quality item.",
}
CLASS_SOURCE = {0: "salminen", 1: "ott", 2: "salminen"}
CLASS_DOMAIN = {0: "product", 1: "hotel", 2: "product"}
CLASS_CATEGORY = {0: "Kitchen", 1: "hilton", 2: "Electronics"}


def tiny_reviews_frame() -> pd.DataFrame:
    """Return a 30-row canonical frame covering all three classes."""
    rows: list[dict[str, Any]] = []
    for label in range(N_CLASSES):
        for index in range(N_PER_CLASS):
            text = f"{CLASS_TEXT[label]} review {label}-{index}"
            rows.append(
                {
                    "id": label * N_PER_CLASS + index,
                    "text": text,
                    "text_clean": text.lower(),
                    "label": label,
                    "source": CLASS_SOURCE[label],
                    "domain": CLASS_DOMAIN[label],
                    "category": CLASS_CATEGORY[label],
                    "group_id": label * N_PER_CLASS + index,
                }
            )
    return pd.DataFrame(rows)


def _write_raw_csvs(raw_dir: Path, frame: pd.DataFrame) -> tuple[Path, Path]:
    """Write tiny Salminen and Ott CSVs in the real column layout."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    salminen_path = raw_dir / "fake_reviews_dataset.csv"
    ott_path = raw_dir / "deceptive-opinion.csv"
    genuine = frame.loc[frame["label"] == 0]
    ai = frame.loc[frame["label"] == 2]
    fake = frame.loc[frame["label"] == 1]
    pd.DataFrame(
        {
            "category": list(genuine["category"]) + list(ai["category"]),
            "rating": [5.0] * len(genuine) + [5.0] * len(ai),
            "label": ["OR"] * len(genuine) + ["CG"] * len(ai),
            "text_": list(genuine["text"]) + list(ai["text"]),
        }
    ).to_csv(salminen_path, index=False)
    pd.DataFrame(
        {
            "deceptive": ["truthful" if label == 0 else "deceptive" for label in fake["label"]],
            "hotel": list(fake["category"]),
            "polarity": ["positive"] * len(fake),
            "source": ["TripAdvisor"] * len(fake),
            "text": list(fake["text"]),
        }
    ).to_csv(ott_path, index=False)
    # Ott needs both truthful (0) and deceptive (1). The 10 class-1 rows are all deceptive.
    # Add 10 truthful hotel rows so load_ott covers both labels.
    extra = pd.DataFrame(
        {
            "deceptive": ["truthful"] * N_PER_CLASS,
            "hotel": ["hilton"] * N_PER_CLASS,
            "polarity": ["positive"] * N_PER_CLASS,
            "source": ["TripAdvisor"] * N_PER_CLASS,
            "text": [f"The room was clean and the staff were helpful stay {index}." for index in range(N_PER_CLASS)],
        }
    )
    ott = pd.read_csv(ott_path)
    pd.concat([extra, ott], ignore_index=True).to_csv(ott_path, index=False)
    return salminen_path, ott_path


def write_tmp_config(tmp_path: Path) -> AttrDict:
    """Write a config.yaml under tmp_path whose paths all stay inside tmp_path."""
    frame = tiny_reviews_frame()
    raw_dir = tmp_path / "data" / "raw"
    salminen_path, ott_path = _write_raw_csvs(raw_dir, frame)
    paths = {
        "raw_dir": tmp_path / "data" / "raw",
        "interim_dir": tmp_path / "data" / "interim",
        "processed_dir": tmp_path / "data" / "processed",
        "splits_dir": tmp_path / "data" / "processed" / "splits",
        "figures_dir": tmp_path / "outputs" / "figures",
        "metrics_dir": tmp_path / "outputs" / "metrics",
        "models_dir": tmp_path / "outputs" / "models",
        "embeddings_dir": tmp_path / "outputs" / "embeddings",
        "logs_dir": tmp_path / "outputs" / "logs",
    }
    payload = {
        "seed": 42,
        "paths": {key: str(value) for key, value in paths.items()},
        "data": {
            "salminen_file": str(salminen_path),
            "ott_file": str(ott_path),
            "genuine_product_sample": 5,
            "ai_sample": 5,
            "min_words": 5,
        },
        "labels": {0: "Genuine Human", 1: "Fake Human-Written", 2: "AI-Generated"},
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        "dedup": {"near_dup_threshold": 0.9},
        "tfidf": {
            "ngram_min": 1,
            "ngram_max": 2,
            "max_features": 300,
            "min_df": 1,
            "sublinear_tf": True,
        },
        "model": {
            "name": "distilbert-base-uncased",
            "max_length": 32,
            "freeze_encoder": True,
            "hidden_units": 16,
            "dropout": 0.0,
            "num_classes": 3,
        },
        "training": {
            "learning_rate": 0.0001,
            "batch_size": 4,
            "epochs": 2,
            "embed_batch_size": 8,
        },
        "experiments": {
            "early_stopping_patience": 2,
            "finetune_learning_rate": 0.00002,
            "finetune_batch_size": 4,
            "finetune_epochs": 1,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    return cfg


def write_tiny_splits(config: AttrDict, frame: pd.DataFrame) -> dict[str, Path]:
    """Write 24/3/3 train/val/test CSVs under the temporary splits directory."""
    splits_dir = Path(config.paths.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("label"):
        ordered = group.reset_index(drop=True)
        train_parts.append(ordered.iloc[:8])
        val_parts.append(ordered.iloc[8:9])
        test_parts.append(ordered.iloc[9:10])
    parts = {
        "train": pd.concat(train_parts, ignore_index=True),
        "val": pd.concat(val_parts, ignore_index=True),
        "test": pd.concat(test_parts, ignore_index=True),
    }
    paths: dict[str, Path] = {}
    for name, part in parts.items():
        path = splits_dir / f"{name}.csv"
        part.to_csv(path, index=False)
        paths[name] = path
    return paths


@pytest.fixture
def tiny_reviews_df() -> pd.DataFrame:
    """30-row dataset with 10 reviews in each class."""
    return tiny_reviews_frame()


@pytest.fixture
def tmp_config(tmp_path: Path) -> AttrDict:
    """Config whose data and output paths all live under tmp_path."""
    return write_tmp_config(tmp_path)


@pytest.fixture
def tmp_config_path(tmp_config: AttrDict) -> Path:
    """Path to the temporary config.yaml written by tmp_config."""
    return Path(tmp_config.config_path)


@pytest.fixture
def tiny_split_paths(tmp_config: AttrDict, tiny_reviews_df: pd.DataFrame) -> dict[str, Path]:
    """Train/val/test CSVs for the 30-row corpus under tmp_config."""
    return write_tiny_splits(tmp_config, tiny_reviews_df)


class FakeTokenizer:
    """Minimal tokenizer used to avoid downloading DistilBERT in tests."""

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 128,
        return_tensors: str | None = "pt",
    ) -> dict[str, Any]:
        """Return padded input_ids and attention_mask tensors."""
        del padding, truncation
        import torch

        batch = len(texts)
        ids = torch.ones(batch, max_length, dtype=torch.long)
        for index, text in enumerate(texts):
            ids[index, 0] = min(len(text.split()), max_length)
        mask = torch.ones(batch, max_length, dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": ids, "attention_mask": mask}
        return {"input_ids": ids.tolist(), "attention_mask": mask.tolist()}


class FakeEncoder:
    """Deterministic encoder that returns [CLS] vectors from token ids."""

    def __init__(self, dim: int = 768) -> None:
        """Store the hidden size used for fake CLS vectors."""
        self.dim = dim

    def eval(self) -> FakeEncoder:
        """No-op eval for API compatibility."""
        return self

    def to(self, device: str) -> FakeEncoder:
        """No-op device move for API compatibility."""
        del device
        return self

    def parameters(self) -> Iterator[Any]:
        """Yield no parameters (frozen fake encoder)."""
        return iter(())

    def __call__(self, **kwargs: Any) -> Any:
        """Return last_hidden_state with a deterministic CLS vector."""
        import torch

        input_ids = kwargs["input_ids"]
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, self.dim)
        hidden[:, 0, 0] = input_ids[:, 0].float()
        hidden[:, 0, 1] = input_ids.float().mean(dim=1)

        class _Out:
            """Container matching Hugging Face model output."""

            last_hidden_state = hidden

        return _Out()


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    """Return a fake DistilBERT tokenizer."""
    return FakeTokenizer()


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    """Return a fake DistilBERT encoder."""
    return FakeEncoder()

"""Tests for frozen DistilBERT [CLS] embedding cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.embed import CLS_DIM, SPLIT_NAMES, embed_texts, embedding_paths, load_encoder

pytestmark = pytest.mark.slow


def _require_embedding_files() -> None:
    """Fail with a clear message if a split's npy cache is missing."""
    cfg = load_config()
    missing = [
        str(path)
        for split in SPLIT_NAMES
        for path in embedding_paths(cfg, split).values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing embedding files (run python -m src.embed first): " + ", ".join(missing)
        )


def test_saved_x_shape_dtype_and_finite() -> None:
    """Each saved X is (n_split, 768) float32 with no NaN or inf."""
    _require_embedding_files()
    cfg = load_config()
    shapes: dict[str, tuple[int, int]] = {}
    for split in SPLIT_NAMES:
        n_rows = len(pd.read_csv(Path(cfg.paths.splits_dir) / f"{split}.csv"))
        X = np.load(embedding_paths(cfg, split)["X"])
        assert X.shape == (n_rows, CLS_DIM), f"{split} X {X.shape} != ({n_rows}, {CLS_DIM})"
        assert X.dtype == np.float32, f"{split} X dtype {X.dtype}"
        assert np.isfinite(X).all(), f"{split} X contains NaN or inf"
        shapes[split] = tuple(X.shape)
        print(f"{split}: X={X.shape} dtype={X.dtype}")
    print(f"shapes={shapes}")


def test_y_and_ids_match_split_csv_row_order() -> None:
    """Saved y and ids match the split CSV labels and ids, in the same row order."""
    _require_embedding_files()
    cfg = load_config()
    for split in SPLIT_NAMES:
        frame = pd.read_csv(Path(cfg.paths.splits_dir) / f"{split}.csv")
        paths = embedding_paths(cfg, split)
        y = np.load(paths["y"])
        ids = np.load(paths["ids"])
        assert np.array_equal(y, frame["label"].astype(int).to_numpy()), f"{split} y mismatch"
        assert np.array_equal(ids, frame["id"].to_numpy()), f"{split} ids mismatch"


@pytest.fixture(scope="module")
def frozen_encoder() -> tuple[Any, Any, str]:
    """Load the frozen DistilBERT encoder once for the slow tests."""
    cfg = load_config()
    encoder, device = load_encoder(cfg)
    return cfg, encoder, device


@pytest.mark.slow
def test_encoder_parameters_require_grad_false(frozen_encoder: tuple[Any, Any, str]) -> None:
    """Every encoder parameter has requires_grad == False."""
    _cfg, encoder, _device = frozen_encoder
    assert all(param.requires_grad is False for param in encoder.parameters())


@pytest.mark.slow
def test_same_review_embedded_twice_matches(frozen_encoder: tuple[Any, Any, str]) -> None:
    """Embedding the same review twice yields max abs difference below 1e-5."""
    cfg, encoder, device = frozen_encoder
    cfg.training.embed_batch_size = 1
    text = ["the hotel room was clean and the staff were helpful"]
    first = embed_texts(text, cfg, encoder, device, desc="repeat-a")
    second = embed_texts(text, cfg, encoder, device, desc="repeat-b")
    max_abs = float(np.max(np.abs(first - second)))
    print(f"same-review max abs diff={max_abs:.2e}")
    assert max_abs < 1e-5


@pytest.mark.slow
def test_padding_mask_keeps_short_review_stable(frozen_encoder: tuple[Any, Any, str]) -> None:
    """A short review alone matches the same review batched with a much longer one."""
    cfg, encoder, device = frozen_encoder
    short = "the hotel was clean"
    long = " ".join(["excellent"] * 80)
    cfg.training.embed_batch_size = 1
    alone = embed_texts([short], cfg, encoder, device, desc="pad-alone")
    cfg.training.embed_batch_size = 2
    batched = embed_texts([short, long], cfg, encoder, device, desc="pad-batch")
    max_abs = float(np.max(np.abs(alone[0] - batched[0])))
    print(f"padding-mask max abs diff={max_abs:.2e}")
    assert max_abs < 1e-4

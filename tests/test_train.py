"""Tests for src.train."""

import numpy as np
import torch

from src.model import build_classifier
from src.train import make_loss, make_optimizer, train_classifier


def test_optimizer_and_loss(cfg: dict) -> None:
    """Config selects Adam and cross-entropy."""
    model = build_classifier(cfg)
    opt = make_optimizer(model, cfg)
    assert opt.__class__.__name__ == "Adam"
    assert abs(opt.param_groups[0]["lr"] - float(cfg["train"]["lr"])) < 1e-12
    assert make_loss(cfg).__class__.__name__ == "CrossEntropyLoss"


def test_train_classifier_runs_epochs(cfg: dict) -> None:
    """Training on synthetic embeddings returns a history with the configured epochs."""
    rng = np.random.default_rng(42)
    dim = int(cfg["model"]["embedding_dim"])
    x_train = rng.normal(size=(24, dim)).astype(np.float32)
    y_train = np.array([0, 1, 2] * 8)
    x_val = rng.normal(size=(6, dim)).astype(np.float32)
    y_val = np.array([0, 1, 2, 0, 1, 2])
    model, history = train_classifier(x_train, y_train, x_val, y_val, cfg)
    assert len(history) == int(cfg["train"]["epochs"])
    assert "train_loss" in history.columns
    logits = model(torch.from_numpy(x_val))
    assert tuple(logits.shape) == (6, 3)

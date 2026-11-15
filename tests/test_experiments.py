"""Tests for class weights, early stopping, fine-tune freezing, and experiments.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import find_project_root, load_config
from src.embed import load_encoder
from src.experiments import (
    N_FINETUNE_LAYERS,
    best_model,
    class_weights,
    prepare_finetune_encoder,
    restore_checkpoint_after_early_stop,
    walk_early_stopping,
)
from src.model import ClassifierHead, count_parameters

EXPERIMENTS_CSV = find_project_root() / "outputs" / "metrics" / "experiments.csv"


def test_class_weights_for_100_50_50_counts() -> None:
    """Inverse-frequency weights for [100, 50, 50] equal [0.6667, 1.3333, 1.3333]."""
    y_train = np.array([0] * 100 + [1] * 50 + [2] * 50, dtype=np.int64)
    weights = class_weights(y_train, 3)
    expected = np.array([0.6667, 1.3333, 1.3333], dtype=np.float32)
    np.testing.assert_allclose(weights.detach().cpu().numpy(), expected, atol=1e-4)


def test_early_stopping_stops_at_epoch_9_and_restores_epoch_4() -> None:
    """Val loss stops improving after epoch 4; patience 5 stops at 9 and restores epoch 4."""
    val_losses = [0.90, 0.80, 0.70, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46]
    checkpoints = [{"epoch": epoch, "weight": float(epoch)} for epoch in range(1, 11)]
    stop_epoch, best_epoch = walk_early_stopping(val_losses, patience=5)
    restored_stop, restored_best, restored = restore_checkpoint_after_early_stop(
        checkpoints, val_losses, patience=5
    )
    assert stop_epoch == 9
    assert best_epoch == 4
    assert restored_stop == 9
    assert restored_best == 4
    assert restored["epoch"] == 4
    assert restored["weight"] == 4.0


@pytest.mark.slow
def test_finetune_unfreezes_only_last_two_layers_and_head() -> None:
    """--finetune: only the last 2 transformer layers and the head have requires_grad."""
    cfg = load_config()
    encoder, _device = load_encoder(cfg)
    encoder = prepare_finetune_encoder(encoder, n_layers=N_FINETUNE_LAYERS)
    head = ClassifierHead(cfg)
    n_layers = len(encoder.transformer.layer)
    last_two = {n_layers - 2, n_layers - 1}
    for name, param in encoder.named_parameters():
        is_last_two = any(f"transformer.layer.{index}." in name for index in last_two)
        assert param.requires_grad is is_last_two, name
    assert all(param.requires_grad for param in head.parameters())
    n_last_two = sum(int(param.numel()) for param in encoder.transformer.layer[-2:].parameters())
    n_head = count_parameters(head, trainable_only=True)
    n_encoder_trainable = count_parameters(encoder, trainable_only=True)
    n_encoder_all = count_parameters(encoder, trainable_only=False)
    print(
        f"finetune trainable: last_2_layers={n_last_two:,} head={n_head:,} "
        f"encoder_trainable={n_encoder_trainable:,} encoder_total={n_encoder_all:,} "
        f"total_trainable={n_last_two + n_head:,}"
    )
    assert n_encoder_trainable == n_last_two
    assert n_head == 197_635
    assert n_last_two > 0


def test_experiments_csv_one_row_per_variant_and_best_is_max_val_macro_f1() -> None:
    """experiments.csv has one row per run variant; best is highest val macro F1."""
    if not EXPERIMENTS_CSV.is_file():
        raise FileNotFoundError(f"Missing {EXPERIMENTS_CSV}. Run python -m src.experiments first.")
    frame = pd.read_csv(EXPERIMENTS_CSV)
    print(frame.to_string(index=False))
    assert frame["variant"].is_unique
    assert len(frame) >= 1
    expected = {"baseline", "class_weighted", "early_stopping", "mean_pooling"}
    assert expected.issubset(set(frame["variant"]))
    winner = best_model(frame.to_dict(orient="records"), "val_macro_f1")
    assert winner is not None
    csv_best = frame.loc[frame["val_macro_f1"].idxmax()]
    assert winner["variant"] == csv_best["variant"]
    assert float(winner["val_macro_f1"]) == pytest.approx(float(frame["val_macro_f1"].max()))

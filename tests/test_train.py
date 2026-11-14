"""Tests for ClassifierHead training: overfit, history, seed, best ckpt, no test split."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.config import find_project_root, load_config
from src.model import ClassifierHead
from src.train import (
    HISTORY_COLUMNS,
    evaluate,
    load_cached_embeddings,
    make_loader,
    train_head,
    train_one_epoch,
)
from src.utils import set_seed

HISTORY_PATH = find_project_root() / "outputs" / "metrics" / "history.csv"
BEST_PATH = find_project_root() / "outputs" / "models" / "head_best.pt"
TRAIN_PY = find_project_root() / "src" / "train.py"


def _real_history() -> pd.DataFrame:
    """Load history.csv from the real 50-epoch run."""
    if not HISTORY_PATH.is_file():
        raise FileNotFoundError(f"Missing {HISTORY_PATH}. Run python -m src.train first.")
    return pd.read_csv(HISTORY_PATH)


def test_overfit_32_embeddings_reaches_perfect_train_accuracy() -> None:
    """300 steps on 32 random embeddings must reach 100% training accuracy."""
    cfg = load_config()
    set_seed(int(cfg.seed))
    rng = np.random.default_rng(int(cfg.seed))
    y = np.array([0, 1, 2, 0] * 8, dtype=np.int64)
    X = rng.normal(size=(32, 768)).astype(np.float32)
    model = ClassifierHead(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(X, y, batch_size=32, shuffle=True, seed=int(cfg.seed))
    eval_loader = make_loader(X, y, batch_size=32, shuffle=False, seed=int(cfg.seed))
    for _ in range(300):
        train_one_epoch(model, loader, criterion, optimizer, "cpu")
    _loss, acc = evaluate(model, eval_loader, criterion, "cpu")
    assert acc == 1.0, f"overfit train_acc={acc} after 300 steps, expected 1.0"


def test_real_history_csv_has_50_epochs_and_required_columns() -> None:
    """Real history.csv has 50 rows, epochs 1..50, and the five required columns."""
    history = _real_history()
    assert list(history.columns) == list(HISTORY_COLUMNS)
    assert len(history) == 50
    assert list(history["epoch"].astype(int)) == list(range(1, 51))
    last5 = history.tail(5)
    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    print(last5.to_string(index=False))
    print(f"best_epoch={best_epoch}")


def test_train_loss_drops_from_epoch_1_to_50() -> None:
    """train_loss at epoch 50 is lower than at epoch 1."""
    history = _real_history()
    loss_1 = float(history.loc[history["epoch"] == 1, "train_loss"].iloc[0])
    loss_50 = float(history.loc[history["epoch"] == 50, "train_loss"].iloc[0])
    assert loss_50 < loss_1, f"train_loss epoch50={loss_50} >= epoch1={loss_1}"


def test_two_cpu_runs_match_for_first_five_epochs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two trainings with seed 42 match for 5 epochs within 1e-6 on CPU."""
    monkeypatch.setattr("src.train.get_device", lambda: "cpu")
    cfg = load_config()
    assert int(cfg.seed) == 42
    cfg.training.epochs = 5
    histories = []
    for name in ("run_a", "run_b"):
        cfg.paths.metrics_dir = tmp_path / name / "metrics"
        cfg.paths.models_dir = tmp_path / name / "models"
        histories.append(train_head(cfg))
    first, second = histories
    for column in HISTORY_COLUMNS:
        np.testing.assert_allclose(
            first[column].to_numpy(dtype=np.float64),
            second[column].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
            err_msg=column,
        )


def test_head_best_reproduces_recorded_val_acc() -> None:
    """Evaluating head_best.pt on val matches history val_acc for its epoch."""
    if not BEST_PATH.is_file():
        raise FileNotFoundError(f"Missing {BEST_PATH}. Run python -m src.train first.")
    cfg = load_config()
    ckpt = torch.load(BEST_PATH, map_location="cpu", weights_only=False)
    epoch = int(ckpt["epoch"])
    head = ClassifierHead(cfg)
    head.load_state_dict(ckpt["state_dict"])
    x_val, y_val = load_cached_embeddings(cfg, "val")
    loader = make_loader(x_val, y_val, int(cfg.training.batch_size), shuffle=False, seed=int(cfg.seed))
    _loss, acc = evaluate(head, loader, nn.CrossEntropyLoss(), "cpu")
    history = _real_history()
    recorded = float(history.loc[history["epoch"] == epoch, "val_acc"].iloc[0])
    assert acc == pytest.approx(recorded, abs=1e-6)


def test_grep_train_py_never_opens_test_split() -> None:
    """grep src/train.py: test-split files are never opened; only refusal comments remain."""
    forbidden = subprocess.run(
        [
            "grep",
            "-nE",
            r'test_X\.npy|test_y\.npy|test_ids\.npy|splits/test|test\.csv|load_cached_embeddings\([^)]*["\']test["\']',
            str(TRAIN_PY),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(forbidden.stdout or "(no forbidden matches)")
    assert forbidden.returncode == 1, f"train.py opens the test split:\n{forbidden.stdout}"
    assert forbidden.stdout.strip() == ""

    mentions = subprocess.run(
        ["grep", "-n", "-i", "test", str(TRAIN_PY)],
        capture_output=True,
        text=True,
        check=False,
    )
    print(mentions.stdout)
    assert mentions.returncode == 0
    for line in mentions.stdout.splitlines():
        lower = line.lower()
        allowed = (
            "must not" in lower
            or "refuse" in lower
            or "train/val" in lower
            or "train or val" in lower
        )
        assert allowed, f"unexpected test mention: {line}"

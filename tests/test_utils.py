"""Tests for src.utils."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import AttrDict, load_config
from src.utils import (
    ensure_dirs,
    get_device,
    get_logger,
    load_json,
    save_json,
    set_seed,
)


def test_set_seed_is_deterministic() -> None:
    """The same seed yields the same NumPy and Torch draws."""
    import torch

    set_seed(42)
    first_np = np.random.rand(3)
    first_th = torch.rand(3)
    set_seed(42)
    second_np = np.random.rand(3)
    second_th = torch.rand(3)
    assert np.allclose(first_np, second_np)
    assert torch.allclose(first_th, second_th)


def test_get_device_known_value() -> None:
    """Device is one of the supported backend names."""
    assert get_device() in {"cuda", "mps", "cpu"}


def test_ensure_dirs_creates_path_entries(tmp_path: Path) -> None:
    """ensure_dirs creates every directory in config.paths."""
    nested = tmp_path / "a" / "b"
    config = AttrDict(paths=AttrDict(one=tmp_path / "one", nested=nested))
    ensure_dirs(config)
    assert (tmp_path / "one").is_dir()
    assert nested.is_dir()


def test_get_logger_writes_run_log() -> None:
    """Logger emits to the console handler and outputs/logs/run.log."""
    cfg = load_config()
    ensure_dirs(cfg)
    logger = get_logger("test_structure_logger")
    logger.info("structure-check")
    log_path = Path(cfg.paths.logs_dir) / "run.log"
    assert log_path.is_file()
    assert "structure-check" in log_path.read_text(encoding="utf-8")


def test_save_and_load_json_numpy_types(tmp_path: Path) -> None:
    """JSON helpers round-trip numpy scalars and arrays."""
    payload = {
        "count": np.int64(3),
        "score": np.float32(1.5),
        "ok": np.bool_(True),
        "vector": np.array([1, 2, 3]),
    }
    path = tmp_path / "out.json"
    save_json(payload, path)
    loaded = load_json(path)
    assert loaded["count"] == 3
    assert loaded["score"] == pytest.approx(1.5)
    assert loaded["ok"] is True
    assert loaded["vector"] == [1, 2, 3]

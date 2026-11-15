"""Combined checks for src.config and src.utils."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from src.config import find_project_root, load_config
from src.utils import get_device, get_logger, load_json, save_json, set_seed

ROOT = find_project_root()


def _raw_config() -> dict[str, Any]:
    """Load the on-disk YAML without resolving or validating extras."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_load_config_from_project_root_and_tests_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_config() resolves the same absolute paths from the root and from tests/."""
    expected_raw = (ROOT / "data" / "raw").resolve()
    expected_salminen = (ROOT / "data" / "raw" / "fake_reviews_dataset.csv").resolve()

    monkeypatch.chdir(ROOT)
    from_root = load_config()
    assert from_root.paths.raw_dir == expected_raw
    assert from_root.data.salminen_file == expected_salminen

    monkeypatch.chdir(ROOT / "tests")
    from_tests = load_config()
    assert from_tests.paths.raw_dir == expected_raw
    assert from_tests.data.salminen_file == expected_salminen
    assert from_tests.paths.logs_dir == from_root.paths.logs_dir


def test_missing_model_section_raises(tmp_path: Path) -> None:
    """A config without the model section raises ValueError naming that section."""
    payload = _raw_config()
    del payload["model"]
    path = tmp_path / "missing_model.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="model"):
        load_config(path)


def test_invalid_split_ratios_raise(tmp_path: Path) -> None:
    """Split ratios 0.7/0.2/0.2 do not sum to 1.0 and must raise ValueError."""
    payload = _raw_config()
    payload["split"] = {"train": 0.7, "val": 0.2, "test": 0.2}
    path = tmp_path / "bad_split.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        load_config(path)


def test_set_seed_repeatable() -> None:
    """set_seed(42) twice reproduces torch, numpy, and random draws."""
    set_seed(42)
    torch_a = torch.rand(5).clone()
    numpy_a = np.random.rand(5).copy()
    py_a = random.random()
    set_seed(42)
    torch_b = torch.rand(5)
    numpy_b = np.random.rand(5)
    py_b = random.random()
    assert torch.equal(torch_a, torch_b)
    assert np.array_equal(numpy_a, numpy_b)
    assert py_a == py_b


def test_json_roundtrip_numpy_types(tmp_path: Path) -> None:
    """save_json then load_json round-trips a numpy float and a numpy int."""
    payload = {"score": np.float64(0.25), "count": np.int64(7)}
    path = tmp_path / "roundtrip.json"
    save_json(payload, path)
    loaded = load_json(path)
    assert loaded["score"] == pytest.approx(0.25)
    assert loaded["count"] == 7
    assert isinstance(loaded["count"], int)


def test_device_and_logger_run_log() -> None:
    """get_device() is a known backend and get_logger writes to outputs/logs/run.log."""
    assert get_device() in {"cuda", "mps", "cpu"}
    logger = get_logger("config_utils_check")
    marker = "config-utils-log-line"
    logger.info(marker)
    log_path = ROOT / "outputs" / "logs" / "run.log"
    assert log_path.is_file()
    assert marker in log_path.read_text(encoding="utf-8")

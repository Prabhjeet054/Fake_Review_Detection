"""Tests for src.utils."""

from pathlib import Path

import pandas as pd

from src.utils import (
    ensure_dir,
    get_device,
    load_dataframe,
    load_json,
    save_dataframe,
    save_json,
    seed_from_cfg,
    set_seed,
    setup_logging,
)


def test_set_seed_is_deterministic() -> None:
    """The same seed yields the same numpy draw."""
    import numpy as np

    set_seed(42)
    first = np.random.rand(3)
    set_seed(42)
    second = np.random.rand(3)
    assert (first == second).all()


def test_json_and_dataframe_roundtrip(tmp_path: Path) -> None:
    """JSON and CSV helpers round-trip data."""
    payload = {"a": 1}
    json_path = save_json(payload, tmp_path / "x.json")
    assert load_json(json_path)["a"] == 1
    frame = pd.DataFrame({"n": [1, 2]})
    csv_path = save_dataframe(frame, tmp_path / "x.csv")
    loaded = load_dataframe(csv_path)
    assert list(loaded["n"]) == [1, 2]


def test_ensure_dir_and_logging(cfg: dict) -> None:
    """Logging creates the configured log file."""
    log_dir = ensure_dir(Path(cfg["_project_root"]) / "outputs" / "logs")
    logger = setup_logging(cfg)
    logger.info("hello")
    assert log_dir.exists()
    assert seed_from_cfg(cfg) == 42
    assert get_device(cfg) in {"cpu", "cuda", "mps"}

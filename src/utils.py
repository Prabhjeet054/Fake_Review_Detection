"""Shared helpers for seeding, logging, IO, and device selection."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, parse_config_args, path_from_cfg


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def setup_logging(cfg: dict[str, Any]) -> logging.Logger:
    """Configure a project logger that writes to stdout and the log file."""
    log_dir = ensure_dir(path_from_cfg(cfg, "paths.logs"))
    log_path = log_dir / cfg["logging"]["filename"]
    logger = logging.getLogger("fake_review_detector")
    logger.setLevel(getattr(logging, str(cfg["logging"]["level"]).upper()))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def save_json(data: Any, path: str | Path) -> Path:
    """Write JSON to disk with UTF-8 encoding."""
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    return out


def load_json(path: str | Path) -> Any:
    """Read JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_joblib(obj: Any, path: str | Path) -> Path:
    """Serialize an object with joblib."""
    out = Path(path)
    ensure_dir(out.parent)
    joblib.dump(obj, out)
    return out


def load_joblib(path: str | Path) -> Any:
    """Load a joblib object."""
    return joblib.load(Path(path))


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame as CSV or parquet based on the suffix."""
    out = Path(path)
    ensure_dir(out.parent)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    return out


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from CSV or parquet."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Data file not found: {src}")
    if src.suffix == ".parquet":
        return pd.read_parquet(src)
    return pd.read_csv(src)


def get_device(cfg: dict[str, Any]) -> str:
    """Select cuda, mps, or cpu from config, with auto-detection."""
    preference = str(cfg["train"]["device"]).lower()
    try:
        import torch
    except ImportError:
        return "cpu"
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_from_cfg(cfg: dict[str, Any]) -> int:
    """Return the configured global seed."""
    return int(cfg["project"]["seed"])


def main(argv: list[str] | None = None) -> None:
    """Apply the configured seed and report the selected device."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    seed = seed_from_cfg(cfg)
    set_seed(seed)
    logger = setup_logging(cfg)
    logger.info("seed=%s device=%s", seed, get_device(cfg))
    print(f"seed={seed}")
    print(f"device={get_device(cfg)}")


if __name__ == "__main__":
    main()

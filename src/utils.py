"""Seeding, device selection, logging, directories, and JSON helpers."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import load_config


def set_seed(seed: int) -> None:
    """Seed random, NumPy, and Torch (CPU and CUDA) and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on what is available."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to the console and to outputs/logs/run.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    cfg = load_config()
    log_dir = Path(cfg.paths.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def ensure_dirs(config: Any) -> None:
    """Create every directory listed in config.paths."""
    paths = config.paths
    values = paths.values() if hasattr(paths, "values") else vars(paths).values()
    for raw in values:
        Path(raw).mkdir(parents=True, exist_ok=True)


def _json_default(obj: Any) -> Any:
    """Convert numpy types (and Path) into JSON-serializable values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(obj: Any, path: str | Path) -> Path:
    """Write JSON to disk, converting numpy types as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)
    return out


def load_json(path: str | Path) -> Any:
    """Read JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> None:
    """Seed RNGs from config, create output dirs, and log the selected device."""
    parser = argparse.ArgumentParser(description="Utilities smoke-check.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    ensure_dirs(cfg)
    logger = get_logger("fake_review_detector")
    device = get_device()
    logger.info("seed=%s device=%s", cfg.seed, device)
    print(f"seed={cfg.seed}")
    print(f"device={device}")


if __name__ == "__main__":
    main()

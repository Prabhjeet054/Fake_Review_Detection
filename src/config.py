"""Load, resolve, and validate config.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = (
    "seed",
    "paths",
    "data",
    "labels",
    "split",
    "dedup",
    "tfidf",
    "model",
    "training",
)

DATA_PATH_KEYS = ("salminen_file", "ott_file")


class AttrDict(dict):
    """Dict that also supports attribute access for nested config values."""

    def __getattr__(self, name: str) -> Any:
        """Return a config value as an attribute."""
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        """Store a config value via attribute assignment."""
        self[name] = value

    def __delattr__(self, name: str) -> None:
        """Delete a config value via attribute deletion."""
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def to_attr_dict(value: Any) -> Any:
    """Recursively convert mappings into AttrDict instances."""
    if isinstance(value, dict):
        return AttrDict({key: to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_attr_dict(item) for item in value]
    return value


def find_project_root() -> Path:
    """Return the project root (the directory that contains src/), independent of cwd."""
    return Path(__file__).resolve().parent.parent


def _resolve_path(root: Path, raw: str | Path) -> Path:
    """Resolve a config path against the project root if it is relative."""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def resolve_paths(cfg: AttrDict, root: Path) -> AttrDict:
    """Resolve every path in cfg.paths and cfg.data relative to the project root."""
    resolved_paths = AttrDict()
    for key, value in cfg.paths.items():
        resolved_paths[key] = _resolve_path(root, value)
    cfg.paths = resolved_paths
    for key in DATA_PATH_KEYS:
        if key in cfg.data:
            cfg.data[key] = _resolve_path(root, cfg.data[key])
    return cfg


def validate_config(cfg: AttrDict) -> None:
    """Raise ValueError if required sections, split ratios, or num_classes are invalid."""
    missing = [section for section in REQUIRED_SECTIONS if section not in cfg]
    if missing:
        raise ValueError(f"Missing required config section(s): {', '.join(missing)}")
    split = cfg.split
    for part in ("train", "val", "test"):
        if part not in split:
            raise ValueError(f"Missing required split key: '{part}'")
    total = float(split.train) + float(split.val) + float(split.test)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0 (got {total})")
    n_labels = len(cfg.labels)
    n_classes = int(cfg.model.num_classes)
    if n_classes != n_labels:
        raise ValueError(
            f"model.num_classes ({n_classes}) must equal the number of labels ({n_labels})"
        )


def load_config(path: str | Path = "config.yaml") -> AttrDict:
    """Load config.yaml as an AttrDict, resolve paths, and validate."""
    root = find_project_root()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("config.yaml must contain a mapping at the top level.")
    cfg = to_attr_dict(raw)
    validate_config(cfg)
    resolve_paths(cfg, root)
    cfg.project_root = root
    cfg.config_path = config_path
    return cfg


def main(argv: list[str] | None = None) -> None:
    """Print the resolved configuration."""
    parser = argparse.ArgumentParser(description="Load and validate config.yaml.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    print(f"seed={cfg.seed}")
    print(f"project_root={cfg.project_root}")
    print(f"raw_dir={cfg.paths.raw_dir}")
    print(f"num_classes={cfg.model.num_classes}")


if __name__ == "__main__":
    main()

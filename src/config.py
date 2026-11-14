"""Load and resolve project configuration from config.yaml."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards until a directory containing config.yaml is found."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start)
    candidates.append(Path(__file__).resolve().parent.parent)
    candidates.append(Path.cwd())
    seen: set[Path] = set()
    for base in candidates:
        current = base.resolve()
        while current not in seen:
            seen.add(current)
            if (current / "config.yaml").exists():
                return current
            if current.parent == current:
                break
            current = current.parent
    raise FileNotFoundError("Could not find config.yaml from the current location.")


def get_nested(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Return a nested value using dotted keys, or default if missing."""
    current: Any = cfg
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def project_root(cfg: dict[str, Any]) -> Path:
    """Return the project root stored on the loaded config."""
    return Path(cfg["_project_root"])


def path_from_cfg(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a config path key (dotted or paths.*) relative to the project root."""
    raw = get_nested(cfg, key) if "." in key else cfg["paths"][key]
    path = Path(raw)
    if not path.is_absolute():
        path = project_root(cfg) / path
    return path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and attach the project root path."""
    if path is None:
        root = find_project_root()
        config_path = root / "config.yaml"
    else:
        config_path = Path(path).expanduser().resolve()
        root = config_path.parent
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must contain a mapping at the top level.")
    cfg = copy.deepcopy(cfg)
    cfg["_project_root"] = str(root)
    cfg["_config_path"] = str(config_path)
    return cfg


def label_names(cfg: dict[str, Any]) -> dict[int, str]:
    """Return {label_id: display name} from config."""
    return {int(key): str(value) for key, value in cfg["labels"].items()}


def dump_config(cfg: dict[str, Any]) -> str:
    """Serialize config for logging, omitting internal keys."""
    public = {key: value for key, value in cfg.items() if not key.startswith("_")}
    return json.dumps(public, indent=2, default=str)


def parse_config_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a shared --config CLI flag."""
    parser = argparse.ArgumentParser(description="Run a fake-review-detector module.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Print the resolved configuration."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    print(dump_config(cfg))
    print(f"project_root={cfg['_project_root']}")


if __name__ == "__main__":
    main()

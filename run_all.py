"""Run the full fake-review-detector pipeline in order."""

from __future__ import annotations

import argparse
from typing import Any, Callable

from src.config import load_config
from src.utils import seed_from_cfg, set_seed, setup_logging

STEP_ORDER = [
    "load",
    "clean",
    "dedup",
    "split",
    "eda",
    "check",
    "features",
    "baselines",
    "embed",
    "train",
    "eval",
    "analysis",
    "experiments",
    "summary",
]


def _call(module_name: str, argv: list[str]) -> None:
    """Import a src module and invoke its main() with the shared CLI args."""
    module = __import__(f"src.{module_name}", fromlist=["main"])
    module.main(argv)


def step_runners() -> dict[str, Callable[[list[str]], None]]:
    """Map pipeline step names to callables."""
    return {
        "load": lambda argv: _call("load_data", argv),
        "clean": lambda argv: _call("clean", argv),
        "dedup": lambda argv: _call("dedup", argv),
        "split": lambda argv: _call("split", argv),
        "eda": lambda argv: _call("eda", argv),
        "check": lambda argv: _call("check_data", argv),
        "features": lambda argv: _call("features", argv),
        "baselines": lambda argv: _call("baselines", argv),
        "embed": lambda argv: _call("embed", argv),
        "train": lambda argv: _call("train", argv),
        "eval": lambda argv: _call("evaluate", argv),
        "analysis": lambda argv: _call("analysis", argv),
        "experiments": lambda argv: _call("experiments", argv),
        "summary": lambda argv: _call("results_summary", argv),
    }


def select_steps(requested: list[str] | None, from_step: str | None) -> list[str]:
    """Choose which pipeline steps to run."""
    if requested:
        unknown = [step for step in requested if step not in STEP_ORDER]
        if unknown:
            raise ValueError(f"Unknown steps: {unknown}. Valid: {STEP_ORDER}")
        return requested
    if from_step:
        if from_step not in STEP_ORDER:
            raise ValueError(f"Unknown from-step '{from_step}'. Valid: {STEP_ORDER}")
        start = STEP_ORDER.index(from_step)
        return STEP_ORDER[start:]
    return list(STEP_ORDER)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse pipeline CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the fake-review-detector pipeline.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--steps", nargs="+", default=None, help="Subset of steps to run")
    parser.add_argument("--from-step", default=None, dest="from_step", help="Start at this step")
    parser.add_argument("--skip", nargs="+", default=None, help="Steps to skip")
    return parser.parse_args(argv)


def run_pipeline(cfg: dict[str, Any], steps: list[str], config_argv: list[str]) -> None:
    """Execute selected steps in order."""
    logger = setup_logging(cfg)
    runners = step_runners()
    for step in steps:
        logger.info("=== step: %s ===", step)
        runners[step](config_argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for `python run_all.py`."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    steps = select_steps(args.steps, args.from_step)
    if args.skip:
        steps = [step for step in steps if step not in set(args.skip)]
    config_argv = ["--config", args.config] if args.config else []
    run_pipeline(cfg, steps, config_argv)


if __name__ == "__main__":
    main()

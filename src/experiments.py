"""Compare saved baseline and deep-model metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import load_config, parse_config_args, path_from_cfg
from src.plots import plot_metrics_comparison
from src.utils import load_json, save_json, seed_from_cfg, set_seed, setup_logging


def load_metric_file(path: Path) -> dict[str, Any] | None:
    """Load a metrics JSON if it exists."""
    if not path.exists():
        return None
    return load_json(path)


def collect_results(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Gather test metrics for baselines and the deep model."""
    metrics_dir = path_from_cfg(cfg, "paths.metrics")
    rows: list[dict[str, Any]] = []
    metric_name = cfg["experiments"]["metric"]
    if cfg["experiments"]["compare_baselines"]:
        for name in cfg["baselines"].keys():
            payload = load_metric_file(metrics_dir / f"baseline_{name}_test.json")
            if payload is None:
                continue
            rows.append(
                {
                    "model": name,
                    "family": "baseline",
                    "accuracy": payload.get("accuracy"),
                    "macro_f1": payload.get("macro_f1"),
                    "weighted_f1": payload.get("weighted_f1"),
                    "cohen_kappa": payload.get("cohen_kappa"),
                    metric_name: payload.get(metric_name),
                }
            )
    if cfg["experiments"]["compare_deep_model"]:
        payload = load_metric_file(metrics_dir / "deep_model_test.json")
        if payload is not None:
            rows.append(
                {
                    "model": "distilbert_mlp",
                    "family": "deep",
                    "accuracy": payload.get("accuracy"),
                    "macro_f1": payload.get("macro_f1"),
                    "weighted_f1": payload.get("weighted_f1"),
                    "cohen_kappa": payload.get("cohen_kappa"),
                    metric_name: payload.get(metric_name),
                }
            )
    return rows


def best_model(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    """Return the row with the highest selected metric."""
    scored = [row for row in rows if row.get(metric) is not None]
    if not scored:
        return None
    return max(scored, key=lambda row: float(row[metric]))


def main(argv: list[str] | None = None) -> None:
    """Write a comparison table of already-computed experiment metrics."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    rows = collect_results(cfg)
    winner = best_model(rows, cfg["experiments"]["metric"])
    payload = {
        "metric": cfg["experiments"]["metric"],
        "n_models": len(rows),
        "best": winner,
        "results": rows,
    }
    out = save_json(payload, path_from_cfg(cfg, "paths.metrics") / "experiments.json")
    if rows:
        plot_metrics_comparison(rows, cfg, "experiment_comparison")
    logger.info("Compared %s models -> %s", len(rows), out)
    print(f"wrote {out} n_models={len(rows)}")
    if winner:
        print(f"best={winner['model']} {cfg['experiments']['metric']}={winner[cfg['experiments']['metric']]}")


if __name__ == "__main__":
    main()

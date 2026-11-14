"""Step 1 structure checks: folders, config keys, and split ratios."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_FOLDERS = [
    "data/raw",
    "data/interim",
    "data/processed",
    "data/processed/splits",
    "src",
    "app",
    "outputs/figures",
    "outputs/metrics",
    "outputs/models",
    "outputs/embeddings",
    "outputs/logs",
    "tests",
]

EXPECTED_CONFIG = {
    "seed": 42,
    "paths": {
        "raw_dir": "data/raw",
        "interim_dir": "data/interim",
        "processed_dir": "data/processed",
        "splits_dir": "data/processed/splits",
        "figures_dir": "outputs/figures",
        "metrics_dir": "outputs/metrics",
        "models_dir": "outputs/models",
        "embeddings_dir": "outputs/embeddings",
        "logs_dir": "outputs/logs",
    },
    "data": {
        "salminen_file": "data/raw/fake_reviews_dataset.csv",
        "ott_file": "data/raw/deceptive-opinion.csv",
        "genuine_product_sample": 1600,
        "ai_sample": 1600,
        "min_words": 5,
    },
    "labels": {
        0: "Genuine Human",
        1: "Fake Human-Written",
        2: "AI-Generated",
    },
    "split": {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15,
    },
    "dedup": {
        "near_dup_threshold": 0.9,
    },
    "tfidf": {
        "ngram_min": 1,
        "ngram_max": 2,
        "max_features": 20000,
        "min_df": 2,
        "sublinear_tf": True,
    },
    "model": {
        "name": "distilbert-base-uncased",
        "max_length": 128,
        "freeze_encoder": True,
        "hidden_units": 256,
        "dropout": 0.3,
        "num_classes": 3,
    },
    "training": {
        "learning_rate": 0.0001,
        "batch_size": 32,
        "epochs": 50,
        "embed_batch_size": 64,
    },
}


def _assert_mapping(actual: dict, expected: dict, prefix: str = "") -> None:
    """Assert every expected key is present with the exact expected value."""
    for key, value in expected.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        assert key in actual, f"Missing config key: {path}"
        if isinstance(value, dict):
            assert isinstance(actual[key], dict), f"{path} must be a mapping"
            _assert_mapping(actual[key], value, path)
        else:
            assert actual[key] == value, f"{path}: expected {value!r}, got {actual[key]!r}"


def test_all_folders_exist() -> None:
    """Every required project folder is present."""
    missing = [folder for folder in EXPECTED_FOLDERS if not (ROOT / folder).is_dir()]
    assert missing == [], f"Missing folders: {missing}"


def test_config_yaml_loads() -> None:
    """config.yaml loads with yaml.safe_load."""
    path = ROOT / "config.yaml"
    assert path.is_file(), "config.yaml is missing"
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert isinstance(cfg, dict), "config.yaml must load as a mapping"


def test_config_keys_match_step1() -> None:
    """Every step-1 config key is present with the exact assigned value."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    _assert_mapping(cfg, EXPECTED_CONFIG)


def test_split_values_sum_to_one() -> None:
    """Train/val/test ratios sum to 1.0."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    total = float(cfg["split"]["train"]) + float(cfg["split"]["val"]) + float(cfg["split"]["test"])
    assert total == 1.0, f"split values sum to {total}, expected 1.0"

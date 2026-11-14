"""Shared pytest fixtures for the fake-review-detector test suite."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from src.config import load_config
from src.schema import add_label_names

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the tests/fixtures directory."""
    return FIXTURES


@pytest.fixture
def cfg(tmp_path: Path) -> dict[str, Any]:
    """Load the real config, retarget paths into a temp project, and shrink hyperparameters."""
    loaded = load_config(PROJECT_ROOT / "config.yaml")
    public = {key: value for key, value in loaded.items() if not str(key).startswith("_")}
    public["train"]["epochs"] = 2
    public["train"]["batch_size"] = 4
    public["train"]["device"] = "cpu"
    public["embed"]["batch_size"] = 4
    public["features"]["tfidf"]["min_df"] = 1
    public["features"]["tfidf"]["max_features"] = 300
    public["features"]["tfidf"]["max_df"] = 1.0
    public["dedup"]["brute_force_limit"] = 1000
    public["dedup"]["method"] = "jaccard"
    public["baselines"]["knn"]["params"] = {"n_neighbors": 3}
    public["baselines"]["decision_tree"]["params"] = {"max_depth": 3}
    public["baselines"]["random_forest"]["params"] = {"n_estimators": 8, "max_depth": 4, "n_jobs": 1}
    public["baselines"]["svm"]["params"] = {"C": 1.0, "max_iter": 2000}
    public["baselines"]["adaboost"]["params"] = {"n_estimators": 8, "algorithm": "SAMME"}
    public["baselines"]["gradient_boosting"]["params"] = {
        "n_estimators": 8,
        "learning_rate": 0.1,
        "max_depth": 2,
    }
    public["explain"]["max_tokens"] = 8
    public["explain"]["top_tokens"] = 4
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(public, sort_keys=False), encoding="utf-8")
    cfg = load_config(config_path)
    (tmp_path / "src" / "resources").mkdir(parents=True, exist_ok=True)
    shutil.copy(PROJECT_ROOT / "src" / "resources" / "contractions.json", tmp_path / "src" / "resources" / "contractions.json")
    raw_sal = tmp_path / "data" / "raw" / "salminen"
    raw_ott = tmp_path / "data" / "raw" / "ott"
    raw_sal.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "salminen_sample.csv", raw_sal / "fake_reviews_dataset.csv")
    shutil.copytree(FIXTURES / "ott", raw_ott, dirs_exist_ok=True)
    for relative in (
        "data/interim",
        "data/processed/splits",
        "outputs/figures",
        "outputs/metrics",
        "outputs/models",
        "outputs/embeddings",
        "outputs/logs",
    ):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.fixture
def config_path(cfg: dict[str, Any]) -> str:
    """Path to the temporary config.yaml."""
    return str(cfg["_config_path"])


@pytest.fixture
def reviews_df(cfg: dict[str, Any]) -> pd.DataFrame:
    """Build a 3-class synthetic corpus large enough for a 70/15/15 split."""
    columns = cfg["columns"]
    rows: list[dict[str, Any]] = []
    templates = {
        0: "I bought this {item} last month and it still works well. The {item} feels sturdy.",
        1: "Wow I cannot believe how perfect this {item} is. Everyone must buy this {item} today.",
        2: "This {item} is amazing. Best {item} ever. Highly recommend this {item} product quality.",
    }
    items = ["blender", "kettle", "lamp", "chair", "phone", "toaster", "mug", "radio", "pillow", "fan", "desk", "router"]
    for label, template in templates.items():
        source = "ott" if label == 1 else "salminen"
        for index, item in enumerate(items):
            text = template.format(item=item) + f" review {label}-{index}"
            rows.append(
                {
                    columns["id"]: f"syn_{label}_{index:02d}",
                    columns["text"]: text,
                    columns["label"]: label,
                    columns["source"]: source,
                    columns["category"]: "product",
                    columns["rating"]: 5.0 if label != 1 else None,
                    columns["polarity"]: None,
                    columns["hotel"]: None,
                }
            )
    # Near-duplicate of the first genuine review.
    first = dict(rows[0])
    first[columns["id"]] = "syn_0_dup"
    first[columns["text"]] = rows[0][columns["text"]]
    rows.append(first)
    return add_label_names(pd.DataFrame(rows), cfg)


class FakeTokenizer:
    """Minimal tokenizer used to avoid downloading DistilBERT in tests."""

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 128,
        return_tensors: str | None = "pt",
    ) -> dict[str, Any]:
        import torch

        batch = len(texts)
        ids = torch.ones(batch, max_length, dtype=torch.long)
        for i, text in enumerate(texts):
            ids[i, 0] = min(len(text.split()), max_length)
        mask = torch.ones(batch, max_length, dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": ids, "attention_mask": mask}
        return {"input_ids": ids.tolist(), "attention_mask": mask.tolist()}


class FakeEncoder:
    """Deterministic encoder that returns [CLS] vectors from token ids."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def eval(self) -> "FakeEncoder":
        """No-op eval for API compatibility."""
        return self

    def to(self, device: str) -> "FakeEncoder":
        """No-op device move for API compatibility."""
        return self

    def parameters(self):
        """Yield no parameters (frozen fake encoder)."""
        return iter(())

    def __call__(self, **kwargs: Any) -> Any:
        import torch

        input_ids = kwargs["input_ids"]
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, self.dim)
        hidden[:, 0, 0] = input_ids[:, 0].float()
        hidden[:, 0, 1] = input_ids.float().mean(dim=1)

        class _Out:
            last_hidden_state = hidden

        return _Out()


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    """Return a fake DistilBERT tokenizer."""
    return FakeTokenizer()


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    """Return a fake DistilBERT encoder."""
    return FakeEncoder()

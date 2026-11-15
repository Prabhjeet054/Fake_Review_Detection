"""Fast tests that exercise pipeline modules on the 30-row tmp fixture."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.baselines import train_baselines
from src.check_data import run_check_data
from src.config import AttrDict
from src.eda import run_eda
from src.embed import (
    CLS_DIM,
    cls_from_hidden,
    embed_split,
    embed_texts,
    embedding_paths,
    split_files_exist,
)
from src.evaluate import evaluate_test
from src.experiments import (
    _mean_embed_split,
    _run_head_experiment,
    _score,
    collect_results,
    mean_embedding_paths,
    run_experiments,
    train_head_variant,
)
from src.load_data import build_dataset
from src.model import (
    ClassifierHead,
    DistilBertClassifier,
    count_parameters,
    print_model_summary,
)
from src.plots import (
    plot_baselines,
    plot_confusion_matrix,
    plot_history,
    plot_per_class_metrics,
)
from src.predict import ReviewClassifier, predict_texts
from src.results_summary import main as results_main
from src.split import (
    _assert_no_group_leakage,
    _fold_to_split,
    split_dataset,
    split_summary,
)
from src.tokenization import tokenize_texts
from src.train import serialize_config, train_head
from src.utils import get_device, set_seed
from tests.conftest import FakeTokenizer


class TinyEncoder(nn.Module):
    """Deterministic nn.Module encoder used instead of DistilBERT."""

    def __init__(self) -> None:
        """Register a dummy parameter so device lookup works."""
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> SimpleNamespace:
        """Return a zero hidden state with a one-hot CLS feature."""
        del attention_mask
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, CLS_DIM, device=input_ids.device, dtype=torch.float32)
        hidden[:, 0, 0] = 1.0
        return SimpleNamespace(last_hidden_state=hidden)


def _write_split_embeddings(config: AttrDict, rng: np.random.Generator) -> None:
    """Write random 768-d embeddings matching each split CSV."""
    Path(config.paths.embeddings_dir).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        frame = pd.read_csv(Path(config.paths.splits_dir) / f"{split}.csv")
        paths = embedding_paths(config, split)
        n_rows = len(frame)
        np.save(paths["X"], rng.normal(size=(n_rows, CLS_DIM)).astype(np.float32))
        np.save(paths["y"], frame["label"].to_numpy(dtype=np.int64))
        np.save(paths["ids"], frame["id"].to_numpy())
        np.save(mean_embedding_paths(config, split), np.load(paths["X"]))


def test_eda_baselines_train_eval_plots_on_tiny_fixture(
    tmp_config: AttrDict,
    tiny_reviews_df: pd.DataFrame,
    tiny_split_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run EDA, baselines, head training, evaluation, and plots on tmp data."""
    del tiny_split_paths
    monkeypatch.setattr("src.train.get_device", lambda: "cpu")
    monkeypatch.setattr("src.evaluate.get_device", lambda: "cpu")
    set_seed(42)
    summary = run_eda(tiny_reviews_df, tmp_config)
    assert summary["n_rows"] == 30
    assert (Path(tmp_config.paths.figures_dir) / "class_distribution.png").is_file()

    tiny_reviews_df.to_csv(Path(tmp_config.paths.processed_dir) / "dataset.csv", index=False)
    issues, card = run_check_data(tmp_config)
    assert isinstance(issues, list)
    assert card.is_file()

    metrics = train_baselines(tmp_config)
    assert not metrics.empty
    assert (Path(tmp_config.paths.metrics_dir) / "baselines.csv").is_file()

    rng = np.random.default_rng(42)
    _write_split_embeddings(tmp_config, rng)
    history = train_head(tmp_config)
    assert len(history) == int(tmp_config.training.epochs)
    assert (Path(tmp_config.paths.models_dir) / "head_best.pt").is_file()

    payload = evaluate_test(tmp_config)
    assert payload["n_test"] == 3
    assert (Path(tmp_config.paths.metrics_dir) / "metrics.json").is_file()

    plot_baselines("test", tmp_config)
    plot_history(tmp_config)
    plot_confusion_matrix(tmp_config)
    plot_per_class_metrics(tmp_config)
    figures = Path(tmp_config.paths.figures_dir)
    assert (figures / "baselines_test.png").is_file()
    assert (figures / "accuracy_curve.png").is_file()
    assert (figures / "confusion_matrix.png").is_file()
    assert (figures / "per_class_metrics.png").is_file()


def test_experiments_and_split_helpers(
    tmp_config: AttrDict,
    tiny_reviews_df: pd.DataFrame,
    tiny_split_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover experiment variants, scoring, and group-aware split helpers."""
    del tiny_split_paths
    monkeypatch.setattr("src.experiments.get_device", lambda: "cpu")
    rng = np.random.default_rng(0)
    _write_split_embeddings(tmp_config, rng)
    row = _run_head_experiment("baseline", tmp_config)
    assert row["variant"] == "baseline"
    assert "val_macro_f1" in row
    y = np.array([0, 1, 2, 0])
    pred = np.array([0, 1, 1, 0])
    scores = _score(y, pred, 3)
    assert 0.0 <= scores["accuracy"] <= 1.0
    model, epochs = train_head_variant(
        tmp_config,
        np.load(embedding_paths(tmp_config, "train")["X"]),
        np.load(embedding_paths(tmp_config, "train")["y"]),
        np.load(embedding_paths(tmp_config, "val")["X"]),
        np.load(embedding_paths(tmp_config, "val")["y"]),
        weighted=True,
        early_stopping=True,
    )
    assert epochs >= 1
    assert isinstance(model, ClassifierHead)
    results = run_experiments(tmp_config, finetune=False, force_mean=False)
    assert "baseline" in set(results["variant"])
    loaded = collect_results(tmp_config)
    assert loaded

    copies = []
    for copy in range(3):
        extra = tiny_reviews_df.copy()
        extra["id"] = extra["id"] + copy * 100
        extra["group_id"] = extra["group_id"] + copy * 100
        extra["text"] = extra["text"] + f" copy{copy}"
        extra["text_clean"] = extra["text_clean"] + f" copy{copy}"
        copies.append(extra)
    fat = pd.concat(copies, ignore_index=True)
    splits = split_dataset(fat, tmp_config)
    assert set(splits) == {"train", "val", "test"}
    _assert_no_group_leakage(splits)
    summary = split_summary(splits)
    assert summary["train"]["n_rows"] > 0
    assert _fold_to_split(0) == "test"
    assert _fold_to_split(3) == "val"
    assert _fold_to_split(10) == "train"


def test_embed_predict_model_tokenize_without_distilbert(
    tmp_config: AttrDict,
    tiny_split_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover embed, predict, model summary, and tokenize via fakes."""
    del tiny_split_paths
    encoder = TinyEncoder()
    tokenizer = FakeTokenizer()
    monkeypatch.setattr("src.embed.tokenize_texts", lambda texts, config: (
        torch.ones(len(texts), int(config.model.max_length), dtype=torch.long),
        torch.ones(len(texts), int(config.model.max_length), dtype=torch.long),
    ))
    monkeypatch.setattr("src.experiments.tokenize_texts", lambda texts, config: (
        torch.ones(len(texts), int(config.model.max_length), dtype=torch.long),
        torch.ones(len(texts), int(config.model.max_length), dtype=torch.long),
    ))
    monkeypatch.setattr("src.tokenization.get_tokenizer", lambda config: tokenizer)
    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1).expand(2, 3, CLS_DIM).clone()
    cls = cls_from_hidden(hidden)
    assert cls.shape == (2, CLS_DIM)
    vectors = embed_texts(["hello world"], tmp_config, encoder, "cpu", desc="tiny")
    assert vectors.shape == (1, CLS_DIM)
    assert split_files_exist(tmp_config, "train") is False
    embed_split(tmp_config, "train", encoder, "cpu", force=True)
    assert split_files_exist(tmp_config, "train") is True
    embed_split(tmp_config, "train", encoder, "cpu", force=False)
    pooled = _mean_embed_split(tmp_config, "train", encoder, "cpu")
    assert pooled.ndim == 2

    ids, mask = tokenize_texts(["a short review"], tmp_config)
    assert ids.shape[0] == 1
    assert mask.shape == ids.shape

    head = ClassifierHead(tmp_config)
    head.eval()
    clf = ReviewClassifier(tmp_config, device="cpu", tokenizer=tokenizer, encoder=encoder, head=head)
    one = clf.predict(["This blender still works well after daily use and cleaning."])
    assert one[0]["label"] in {0, 1, 2}
    assert abs(sum(one[0]["probabilities"].values()) - 1.0) < 1e-5
    batch = predict_texts(
        ["This blender still works well after daily use and cleaning."],
        tmp_config,
        tokenizer=tokenizer,
        encoder=encoder,
        head=head,
        device="cpu",
    )
    assert len(batch) == 1

    wrapped = DistilBertClassifier(tmp_config, encoder=encoder)
    logits = wrapped(
        torch.zeros(2, int(tmp_config.model.max_length), dtype=torch.long),
        torch.ones(2, int(tmp_config.model.max_length), dtype=torch.long),
    )
    assert logits.shape == (2, 3)
    assert count_parameters(wrapped, trainable_only=True) > 0
    print_model_summary(tmp_config, model=wrapped)
    assert serialize_config(tmp_config)["seed"] == 42
    assert get_device() in {"cuda", "mps", "cpu"}


def test_build_dataset_and_results_summary_cli(
    tmp_config: AttrDict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build a tiny merged dataset and write the results summary from tmp metrics."""
    merged = build_dataset(tmp_config)
    assert merged["label"].nunique() == 3
    assert (Path(tmp_config.paths.interim_dir) / "merged.csv").is_file()
    pd.DataFrame(
        {"variant": ["baseline"], "val_macro_f1": [0.5], "test_macro_f1": [0.4]}
    ).to_csv(Path(tmp_config.paths.metrics_dir) / "experiments.csv", index=False)
    monkeypatch.setattr("src.results_summary.load_config", lambda path=None: tmp_config)
    results_main([])
    assert (Path(tmp_config.paths.metrics_dir) / "results_summary.md").is_file()

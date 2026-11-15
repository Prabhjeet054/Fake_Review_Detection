"""Run seeded classifier-head and optional encoder fine-tuning experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import AttrDict, load_config
from src.embed import embedding_paths, load_encoder
from src.model import ClassifierHead
from src.tokenization import tokenize_texts
from src.utils import get_device, set_seed


VARIANT_COLUMNS = (
    "variant",
    "val_accuracy",
    "val_macro_f1",
    "val_genuine_precision",
    "val_genuine_recall",
    "test_accuracy",
    "test_macro_f1",
    "test_genuine_precision",
    "test_genuine_recall",
    "epochs_trained",
)


def load_metric_file(path: Path) -> dict[str, Any] | None:
    """Load a metrics JSON if it exists."""
    import json

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_model(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    """Return the row with the highest selected metric."""
    scored = [row for row in rows if row.get(metric) is not None]
    return max(scored, key=lambda row: float(row[metric])) if scored else None


def collect_results(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load rows from the experiment CSV when it exists."""
    path = Path(cfg["paths"]["metrics_dir"]) / "experiments.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def mean_embedding_paths(config: AttrDict, split: str) -> Path:
    """Return the cache path for mean-pooled embeddings."""
    return Path(config.paths.embeddings_dir) / f"mean_{split}_X.npy"


def _split_frame(config: AttrDict, split: str) -> pd.DataFrame:
    """Load one canonical split frame."""
    path = Path(config.paths.splits_dir) / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}. Run python -m src.split first.")
    return pd.read_csv(path)


def _load_embeddings(config: AttrDict, split: str, mean: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load one split's embeddings and labels."""
    if mean:
        x_path = mean_embedding_paths(config, split)
        if not x_path.is_file():
            raise FileNotFoundError(f"Missing mean embedding cache: {x_path}")
        X = np.load(x_path).astype(np.float32, copy=False)
        y = _split_frame(config, split)["label"].astype(int).to_numpy()
        return X, y
    paths = embedding_paths(config, split)
    if not paths["X"].is_file():
        raise FileNotFoundError(f"Missing embedding file: {paths['X']}. Run python -m src.embed first.")
    X = np.load(paths["X"]).astype(np.float32, copy=False)
    y = _split_frame(config, split)["label"].astype(np.int64).to_numpy()
    if len(X) != len(y):
        raise ValueError(f"Embedding/label count mismatch for {split}: {len(X)} != {len(y)}")
    return X, y


def _loader(X: np.ndarray, y: np.ndarray, config: AttrDict, shuffle: bool) -> DataLoader:
    """Create a deterministic embedding DataLoader."""
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32, copy=False)),
        torch.from_numpy(y.astype(np.int64, copy=False)),
    )
    generator = torch.Generator().manual_seed(int(config.seed))
    return DataLoader(
        dataset,
        batch_size=int(config.training.batch_size),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    """Return inverse-frequency weights n/(C*count) from the training labels."""
    counts = np.bincount(y_train.astype(int), minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError("Every class must be present in the training split.")
    return torch.tensor(
        len(y_train) / (n_classes * counts),
        dtype=torch.float32,
    )


def early_stopping_step(
    val_loss: float,
    best_loss: float,
    stale: int,
    patience: int,
) -> tuple[float, int, bool, bool]:
    """Update early-stopping state; return best_loss, stale, improved, should_stop."""
    if float(val_loss) < float(best_loss):
        return float(val_loss), 0, True, False
    stale += 1
    return float(best_loss), stale, False, stale >= int(patience)


def walk_early_stopping(val_losses: Sequence[float], patience: int) -> tuple[int, int]:
    """Walk a 1-indexed val-loss sequence and return (stop_epoch, best_epoch)."""
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    stop_epoch = len(val_losses)
    for epoch, loss in enumerate(val_losses, start=1):
        best_loss, stale, improved, should_stop = early_stopping_step(
            float(loss), best_loss, stale, patience
        )
        if improved:
            best_epoch = epoch
        if should_stop:
            return epoch, best_epoch
    return stop_epoch, best_epoch


def restore_checkpoint_after_early_stop(
    checkpoints: Sequence[dict[str, Any]],
    val_losses: Sequence[float],
    patience: int,
) -> tuple[int, int, dict[str, Any]]:
    """Stop after `patience` non-improving epochs and restore the best checkpoint."""
    if len(checkpoints) != len(val_losses):
        raise ValueError("checkpoints and val_losses must have the same length.")
    stop_epoch, best_epoch = walk_early_stopping(val_losses, patience)
    return stop_epoch, best_epoch, checkpoints[best_epoch - 1]


N_FINETUNE_LAYERS = 2


def prepare_finetune_encoder(encoder: nn.Module, n_layers: int = N_FINETUNE_LAYERS) -> nn.Module:
    """Freeze the encoder, then unfreeze the last n transformer layers."""
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    layers = encoder.transformer.layer[-int(n_layers) :]
    for layer in layers:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    return encoder


@torch.no_grad()
def _predict_head(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    config: AttrDict,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return predictions, probabilities, and mean cross-entropy loss."""
    loader = _loader(X, y, config, shuffle=False)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    total_loss = 0.0
    total = 0
    for features, labels in loader:
        logits = model(features.to(device))
        loss = criterion(logits, labels.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        predictions.append(probs.argmax(axis=1))
        probabilities.append(probs)
        total_loss += float(loss.item()) * len(labels)
        total += len(labels)
    return (
        np.concatenate(predictions),
        np.vstack(probabilities),
        total_loss / total if total else 0.0,
    )


def _score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    """Calculate accuracy, macro F1, and Genuine-class precision/recall."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "genuine_precision": float(precision[0]),
        "genuine_recall": float(recall[0]),
    }


def train_head_variant(
    config: AttrDict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    weighted: bool = False,
    early_stopping: bool = False,
) -> tuple[nn.Module, int]:
    """Train one deterministic head variant and restore its best validation weights."""
    set_seed(int(config.seed))
    device = get_device()
    model = ClassifierHead(config).to(device)
    weights = class_weights(y_train, int(config.model.num_classes)).to(device) if weighted else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training.learning_rate))
    train_loader = _loader(X_train, y_train, config, shuffle=True)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    trained = 0
    patience = int(config.experiments.early_stopping_patience)
    for epoch in range(1, int(config.training.epochs) + 1):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        _, _, val_loss = _predict_head(model, X_val, y_val, config, device)
        trained = epoch
        best_loss, stale, improved, should_stop = early_stopping_step(
            val_loss, best_loss, stale, patience
        )
        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if early_stopping and should_stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return model, trained


def _mean_embed_split(
    config: AttrDict,
    split: str,
    encoder: nn.Module,
    device: str,
) -> np.ndarray:
    """Mean-pool non-padding token embeddings for one split."""
    frame = _split_frame(config, split)
    texts = frame["text_clean"].fillna("").astype(str).tolist()
    batch_size = int(config.training.embed_batch_size)
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        input_ids, mask = tokenize_texts(texts[start : start + batch_size], config)
        with torch.no_grad():
            hidden = encoder(
                input_ids=input_ids.to(device),
                attention_mask=mask.to(device),
            ).last_hidden_state
            weights = mask.to(device).unsqueeze(-1).float()
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        chunks.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(chunks) if chunks else np.zeros((0, 768), dtype=np.float32)


def build_mean_embedding_cache(config: AttrDict, force: bool = False) -> None:
    """Build mean-pooled embedding caches for train, validation, and test."""
    paths = [mean_embedding_paths(config, split) for split in ("train", "val", "test")]
    if not force and all(path.is_file() for path in paths):
        return
    encoder, device = load_encoder(config)
    for split, path in zip(("train", "val", "test"), paths):
        if path.is_file() and not force:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, _mean_embed_split(config, split, encoder, device))


def _run_head_experiment(
    name: str,
    config: AttrDict,
    mean: bool = False,
    weighted: bool = False,
    early_stopping: bool = False,
) -> dict[str, Any]:
    """Train and score one frozen-encoder head variant."""
    X_train, y_train = _load_embeddings(config, "train", mean)
    X_val, y_val = _load_embeddings(config, "val", mean)
    X_test, y_test = _load_embeddings(config, "test", mean)
    model, epochs = train_head_variant(
        config, X_train, y_train, X_val, y_val, weighted=weighted, early_stopping=early_stopping
    )
    device = get_device()
    val_pred, _, _ = _predict_head(model, X_val, y_val, config, device)
    test_pred, _, _ = _predict_head(model, X_test, y_test, config, device)
    val = _score(y_val, val_pred, int(config.model.num_classes))
    test = _score(y_test, test_pred, int(config.model.num_classes))
    return {
        "variant": name,
        **{f"val_{key}": value for key, value in val.items()},
        **{f"test_{key}": value for key, value in test.items()},
        "epochs_trained": epochs,
    }


def _run_finetune(config: AttrDict) -> dict[str, Any]:
    """Fine-tune the top two DistilBERT layers and the head on a GPU."""
    device = get_device()
    if device not in {"cuda", "mps"}:
        raise RuntimeError("Fine-tuning requires a GPU; rerun with --finetune on CUDA or MPS.")
    set_seed(int(config.seed))
    encoder, _ = load_encoder(config)
    encoder = prepare_finetune_encoder(encoder, n_layers=N_FINETUNE_LAYERS)
    head = ClassifierHead(config).to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in list(encoder.parameters()) + list(head.parameters()) if parameter.requires_grad],
        lr=float(config.experiments.finetune_learning_rate),
    )
    criterion = nn.CrossEntropyLoss()
    frames = {split: _split_frame(config, split) for split in ("train", "val", "test")}
    batch_size = int(config.experiments.finetune_batch_size)
    best_state: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None = None
    best_loss = float("inf")
    for _ in range(1, int(config.experiments.finetune_epochs) + 1):
        encoder.train()
        head.train()
        frame = frames["train"]
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start : start + batch_size]
            input_ids, mask = tokenize_texts(batch["text_clean"].tolist(), config)
            labels = torch.tensor(batch["label"].to_numpy(), dtype=torch.long, device=device)
            optimizer.zero_grad()
            hidden = encoder(input_ids=input_ids.to(device), attention_mask=mask.to(device)).last_hidden_state
            loss = criterion(head(hidden[:, 0, :]), labels)
            loss.backward()
            optimizer.step()
        encoder.eval()
        head.eval()
        with torch.no_grad():
            input_ids, mask = tokenize_texts(frames["val"]["text_clean"].tolist(), config)
            hidden = encoder(input_ids=input_ids.to(device), attention_mask=mask.to(device)).last_hidden_state
            val_loss = float(criterion(
                head(hidden[:, 0, :]),
                torch.tensor(frames["val"]["label"].to_numpy(), dtype=torch.long, device=device),
            ).item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = (
                {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()},
                {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
            )
    if best_state is not None:
        encoder.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])

    def score(frame: pd.DataFrame) -> dict[str, float]:
        """Score the fine-tuned encoder and head on one split."""
        with torch.no_grad():
            input_ids, mask = tokenize_texts(frame["text_clean"].tolist(), config)
            hidden = encoder(input_ids=input_ids.to(device), attention_mask=mask.to(device)).last_hidden_state
            pred = head(hidden[:, 0, :]).argmax(dim=1).cpu().numpy()
        return _score(frame["label"].to_numpy(), pred, int(config.model.num_classes))

    val = score(frames["val"])
    test = score(frames["test"])
    return {
        "variant": "finetune_top2",
        **{f"val_{key}": value for key, value in val.items()},
        **{f"test_{key}": value for key, value in test.items()},
        "epochs_trained": int(config.experiments.finetune_epochs),
    }


def run_experiments(config: AttrDict | None = None, finetune: bool = False, force_mean: bool = False) -> pd.DataFrame:
    """Run all requested variants and write CSV plus markdown results."""
    cfg = config if config is not None else load_config()
    build_mean_embedding_cache(cfg, force=force_mean)
    rows = [
        _run_head_experiment("baseline", cfg),
        _run_head_experiment("class_weighted", cfg, weighted=True),
        _run_head_experiment("early_stopping", cfg, early_stopping=True),
        _run_head_experiment("mean_pooling", cfg, mean=True),
    ]
    if finetune:
        rows.append(_run_finetune(cfg))
    results = pd.DataFrame(rows, columns=list(VARIANT_COLUMNS))
    metrics_dir = Path(cfg.paths.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "experiments.csv"
    results.to_csv(csv_path, index=False)
    try:
        table = results.to_markdown(index=False)
    except ImportError:
        table = results.to_string(index=False)
    markdown = ["# Experiment comparison", "", table, ""]
    (metrics_dir / "experiments.md").write_text("\n".join(markdown), encoding="utf-8")
    winner = results.loc[results["val_macro_f1"].idxmax()]
    print(f"wrote {csv_path}")
    print(f"best_on_val_macro_f1={winner['variant']} ({winner['val_macro_f1']:.4f})")
    return results


def main(argv: list[str] | None = None) -> None:
    """CLI: run frozen-head comparisons and optional GPU fine-tuning."""
    parser = argparse.ArgumentParser(description="Compare fake-review classifier experiments.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--finetune", action="store_true", help="Run optional GPU fine-tuning.")
    parser.add_argument("--force-mean-cache", action="store_true", help="Rebuild mean embedding caches.")
    args = parser.parse_args(argv)
    run_experiments(load_config(args.config), finetune=args.finetune, force_mean=args.force_mean_cache)


if __name__ == "__main__":
    main()

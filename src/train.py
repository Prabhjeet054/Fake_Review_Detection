# The test split must not be read by any training code.
"""Train ClassifierHead on cached DistilBERT [CLS] embeddings (train/val only)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import AttrDict, load_config
from src.embed import embedding_paths
from src.model import ClassifierHead
from src.utils import ensure_dirs, get_device, get_logger, set_seed

ALLOWED_SPLITS = frozenset({"train", "val"})
HISTORY_COLUMNS = ("epoch", "train_loss", "train_acc", "val_loss", "val_acc")


def serialize_config(config: AttrDict) -> dict[str, Any]:
    """Convert config to a plain JSON-like dict (paths become strings)."""

    def convert(value: Any) -> Any:
        """Recursively convert AttrDict values and Paths into JSON-safe objects."""
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return convert(dict(config))


def load_cached_embeddings(config: AttrDict, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load X and y npy caches for train or val. Refuses the test split."""
    if split not in ALLOWED_SPLITS:
        raise RuntimeError("The test split must not be loaded by training code.")
    paths = embedding_paths(config, split)
    x_path, y_path = paths["X"], paths["y"]
    if not x_path.is_file():
        raise FileNotFoundError(f"Missing embedding file: {x_path}. Run python -m src.embed first.")
    if not y_path.is_file():
        raise FileNotFoundError(f"Missing embedding file: {y_path}. Run python -m src.embed first.")
    X = np.load(x_path).astype(np.float32, copy=False)
    y = np.load(y_path).astype(np.int64, copy=False)
    return X, y


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Wrap embeddings in a TensorDataset DataLoader; shuffle uses a seeded generator."""
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
    )


def save_checkpoint(model: nn.Module, config: AttrDict, epoch: int, path: Path) -> Path:
    """Save state_dict, the config used, and the epoch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "config": serialize_config(config),
        "epoch": int(epoch),
    }
    torch.save(payload, path)
    return path


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> None:
    """Run one shuffled training epoch (dropout on). Metrics are not recorded here."""
    model.train()
    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Mean loss and accuracy in eval mode over every batch in loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        logits = model(embeddings)
        loss = criterion(logits, labels)
        batch = int(labels.size(0))
        total_loss += float(loss.item()) * batch
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        n += batch
    if n == 0:
        return 0.0, 0.0
    return total_loss / n, correct / n


def train_head(config: AttrDict | None = None) -> pd.DataFrame:
    """Train ClassifierHead on train/val embeddings and write history plus checkpoints."""
    cfg = config if config is not None else load_config()
    set_seed(int(cfg.seed))
    ensure_dirs(cfg)
    logger = get_logger(__name__)
    device = get_device()

    x_train, y_train = load_cached_embeddings(cfg, "train")
    x_val, y_val = load_cached_embeddings(cfg, "val")
    batch_size = int(cfg.training.batch_size)
    seed = int(cfg.seed)
    train_loader = make_loader(x_train, y_train, batch_size, shuffle=True, seed=seed)
    train_eval_loader = make_loader(x_train, y_train, batch_size, shuffle=False, seed=seed)
    val_loader = make_loader(x_val, y_val, batch_size, shuffle=False, seed=seed)

    model = ClassifierHead(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.training.learning_rate))
    criterion = nn.CrossEntropyLoss()

    models_dir = Path(cfg.paths.models_dir)
    best_path = models_dir / "head_best.pt"
    last_path = models_dir / "head_last.pt"
    history_path = Path(cfg.paths.metrics_dir) / "history.csv"

    epochs = int(cfg.training.epochs)
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    rows: list[dict[str, Any]] = []

    logger.info(
        "Training ClassifierHead on device=%s n_train=%s n_val=%s epochs=%s batch=%s lr=%s",
        device,
        len(y_train),
        len(y_val),
        epochs,
        batch_size,
        cfg.training.learning_rate,
    )
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_loss, train_acc = evaluate(model, train_eval_loader, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
        rows.append(row)
        logger.info(
            "epoch=%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            save_checkpoint(model, cfg, epoch, best_path)

    save_checkpoint(model, cfg, epochs, last_path)
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    history = pd.DataFrame(rows, columns=list(HISTORY_COLUMNS))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path, index=False)
    logger.info(
        "Best epoch=%s val_loss=%.4f -> %s ; last -> %s ; history -> %s",
        best_epoch,
        best_val_loss,
        best_path,
        last_path,
        history_path,
    )
    print(f"best_epoch={best_epoch} best_val_loss={best_val_loss:.4f} wrote {best_path}")
    print(f"wrote {last_path}")
    print(f"wrote {history_path}")
    return history


def main(argv: list[str] | None = None) -> None:
    """CLI: train the classification head on cached train/val embeddings."""
    parser = argparse.ArgumentParser(description="Train ClassifierHead on frozen DistilBERT embeddings.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args(argv)
    train_head(load_config(args.config))


if __name__ == "__main__":
    main()

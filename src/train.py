"""Train the classification head on cached DistilBERT embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config, parse_config_args, path_from_cfg
from src.embed import load_embeddings
from src.evaluate import compute_metrics
from src.model import build_classifier, save_classifier
from src.plots import plot_training_curves
from src.utils import get_device, save_dataframe, seed_from_cfg, set_seed, setup_logging


def make_loader(embeddings: np.ndarray, labels: np.ndarray, cfg: dict[str, Any], *, shuffle: bool) -> DataLoader:
    """Wrap embedding arrays in a Torch DataLoader."""
    x = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    dataset = TensorDataset(x, y)
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["train"]["num_workers"]),
    )


def make_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """Build the configured optimizer (Adam by default)."""
    name = str(cfg["train"]["optimizer"]).lower()
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{name}'.")


def make_loss(cfg: dict[str, Any]) -> nn.Module:
    """Build the configured training loss."""
    name = str(cfg["train"]["loss"]).lower()
    if name in {"cross_entropy", "crossentropy", "ce"}:
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss '{name}'.")


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Batch accuracy from class logits."""
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


@torch.no_grad()
def run_epoch_eval(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate loss/accuracy and collect predictions."""
    model.eval()
    total_loss = 0.0
    n = 0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        logits = model(embeddings)
        loss = criterion(logits, labels)
        batch = labels.size(0)
        total_loss += float(loss.item()) * batch
        n += batch
        y_true.append(labels.detach().cpu().numpy())
        y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
    true_arr = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    pred_arr = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)
    acc = float((true_arr == pred_arr).mean()) if n else 0.0
    return total_loss / max(n, 1), acc, true_arr, pred_arr


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """Run one training epoch and return mean loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch = labels.size(0)
        total_loss += float(loss.item()) * batch
        total_acc += accuracy_from_logits(logits.detach(), labels) * batch
        n += batch
    return total_loss / max(n, 1), total_acc / max(n, 1)


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    cfg: dict[str, Any],
) -> tuple[nn.Module, pd.DataFrame]:
    """Train the MLP head and return the model plus history."""
    set_seed(seed_from_cfg(cfg))
    device = get_device(cfg)
    model = build_classifier(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    criterion = make_loss(cfg)
    train_loader = make_loader(x_train, y_train, cfg, shuffle=True)
    val_loader = None if x_val is None else make_loader(x_val, y_val, cfg, shuffle=False)
    patience = cfg["train"]["patience"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    bad_epochs = 0
    history_rows: list[dict[str, Any]] = []
    epochs = int(cfg["train"]["epochs"])
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        row: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}
        if val_loader is not None:
            val_loss, val_acc, _, _ = run_epoch_eval(model, val_loader, criterion, device)
            row["val_loss"] = val_loss
            row["val_acc"] = val_acc
            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
        history_rows.append(row)
        if patience is not None and val_loader is not None and bad_epochs >= int(patience):
            break
    if x_val is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, pd.DataFrame(history_rows)


def main(argv: list[str] | None = None) -> None:
    """Train on cached embeddings and write the classifier checkpoint."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    x_train, y_train, _ = load_embeddings(cfg, "train")
    try:
        x_val, y_val, _ = load_embeddings(cfg, "val")
    except FileNotFoundError:
        x_val, y_val = None, None
    model, history = train_classifier(x_train, y_train, x_val, y_val, cfg)
    history_path = save_dataframe(history, path_from_cfg(cfg, "paths.logs") / cfg["train"]["log_filename"])
    plot_training_curves(history, cfg, "training_curves")
    extra: dict[str, Any] = {"history_path": str(history_path)}
    if x_val is not None:
        device = get_device(cfg)
        loader = make_loader(x_val, y_val, cfg, shuffle=False)
        _, _, y_true, y_pred = run_epoch_eval(model, loader, make_loss(cfg), device)
        extra["val_metrics"] = compute_metrics(y_true, y_pred, cfg)
    path = save_classifier(model, cfg, extra=extra)
    logger.info("Saved classifier -> %s", path)
    print(f"wrote {path} epochs={len(history)}")


if __name__ == "__main__":
    main()

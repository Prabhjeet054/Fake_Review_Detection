"""Classification head over frozen DistilBERT [CLS] embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.config import load_config, parse_config_args, path_from_cfg
from src.utils import seed_from_cfg, set_seed


class ReviewClassifier(nn.Module):
    """Linear(embedding_dim, hidden) -> ReLU -> Dropout -> Linear(hidden, num_classes)."""

    def __init__(self, embedding_dim: int, hidden_size: int, dropout: float, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of embeddings."""
        hidden = self.activation(self.fc1(embeddings))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


def build_classifier(cfg: dict[str, Any]) -> ReviewClassifier:
    """Construct the review classifier from config hyperparameters."""
    spec = cfg["model"]
    return ReviewClassifier(
        embedding_dim=int(spec["embedding_dim"]),
        hidden_size=int(spec["hidden_size"]),
        dropout=float(spec["dropout"]),
        num_classes=int(spec["num_classes"]),
    )


def classifier_path(cfg: dict[str, Any]) -> Path:
    """Return the configured checkpoint path for the classifier head."""
    return path_from_cfg(cfg, "paths.models") / cfg["model"]["classifier_filename"]


def save_classifier(model: ReviewClassifier, cfg: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    """Save classifier weights and hyperparameters."""
    path = classifier_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "model_config": {
            "embedding_dim": int(cfg["model"]["embedding_dim"]),
            "hidden_size": int(cfg["model"]["hidden_size"]),
            "dropout": float(cfg["model"]["dropout"]),
            "num_classes": int(cfg["model"]["num_classes"]),
        },
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_classifier(cfg: dict[str, Any], map_location: str | None = None) -> ReviewClassifier:
    """Load a previously saved classifier head."""
    path = classifier_path(cfg)
    try:
        payload = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location or "cpu")
    spec = payload.get("model_config", cfg["model"])
    model = ReviewClassifier(
        embedding_dim=int(spec["embedding_dim"]),
        hidden_size=int(spec["hidden_size"]),
        dropout=float(spec["dropout"]),
        num_classes=int(spec["num_classes"]),
    )
    model.load_state_dict(payload["state_dict"])
    return model


def main(argv: list[str] | None = None) -> None:
    """Instantiate the classifier and print parameter counts."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    model = build_classifier(cfg)
    n_params = sum(param.numel() for param in model.parameters())
    print(model)
    print(f"parameters={n_params}")


if __name__ == "__main__":
    main()

"""Tests for src.model."""

import torch

from src.model import ReviewClassifier, build_classifier, load_classifier, save_classifier


def test_forward_shape(cfg: dict) -> None:
    """The head maps (batch, 768) to (batch, 3)."""
    model = build_classifier(cfg)
    batch = torch.randn(5, int(cfg["model"]["embedding_dim"]))
    logits = model(batch)
    assert tuple(logits.shape) == (5, int(cfg["model"]["num_classes"]))
    assert isinstance(model, ReviewClassifier)


def test_save_and_load(cfg: dict) -> None:
    """Classifier weights round-trip through the checkpoint file."""
    model = build_classifier(cfg)
    with torch.no_grad():
        model.fc1.bias.fill_(0.25)
    path = save_classifier(model, cfg)
    assert path.exists()
    loaded = load_classifier(cfg, map_location="cpu")
    assert torch.allclose(loaded.fc1.bias, model.fc1.bias)

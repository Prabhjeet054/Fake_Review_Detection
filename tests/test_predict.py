"""Tests for src.predict."""

from unittest.mock import patch

import numpy as np
import torch

from src.model import build_classifier, save_classifier
from src.predict import predict_texts, softmax_probs


def test_softmax_rows_sum_to_one() -> None:
    """Softmax probabilities are a valid distribution."""
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax_probs(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_predict_texts_with_fakes(cfg: dict, fake_tokenizer, fake_encoder) -> None:
    """End-to-end prediction works with a saved head and fake encoder."""
    model = build_classifier(cfg)
    save_classifier(model, cfg)
    with patch("src.predict.load_tokenizer", return_value=fake_tokenizer), patch(
        "src.predict.load_encoder", return_value=fake_encoder
    ):
        results = predict_texts(["This blender is great and I use it daily."], cfg, device="cpu")
    assert len(results) == 1
    assert results[0]["label"] in {0, 1, 2}
    assert abs(sum(results[0]["probabilities"].values()) - 1.0) < 1e-5
    assert results[0]["text_clean"]

"""Tests for src.embed."""

import numpy as np

from src.embed import cls_from_hidden, encode_texts, load_embeddings, save_embeddings


def test_cls_from_hidden_selects_first_token() -> None:
    """The CLS vector is the first position of the last hidden state."""
    import torch

    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
    cls = cls_from_hidden(hidden)
    assert tuple(cls.shape) == (2, 3)
    assert torch.equal(cls[0], hidden[0, 0])


def test_encode_and_cache(cfg: dict, fake_tokenizer, fake_encoder) -> None:
    """Embeddings are (n, 768) and round-trip through the cache."""
    texts = ["one review", "second review", "third"]
    vectors = encode_texts(texts, fake_tokenizer, fake_encoder, cfg, device="cpu")
    dim = int(cfg["model"]["embedding_dim"])
    assert vectors.shape == (3, dim)
    labels = np.array([0, 1, 2])
    save_embeddings(vectors, labels, ["a", "b", "c"], cfg, "train")
    loaded, loaded_labels, ids = load_embeddings(cfg, "train")
    assert loaded.shape == vectors.shape
    assert np.array_equal(loaded_labels, labels)
    assert ids == ["a", "b", "c"]

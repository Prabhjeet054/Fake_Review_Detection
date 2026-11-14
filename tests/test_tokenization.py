"""Tests for src.tokenization."""

from src.tokenization import token_ids_numpy, tokenize_texts


def test_tokenize_texts_shape(cfg: dict, fake_tokenizer) -> None:
    """Tokenization pads to the configured max length."""
    texts = ["hello world", "another review"]
    encoded = tokenize_texts(texts, fake_tokenizer, cfg, return_tensors="pt")
    max_len = int(cfg["model"]["max_length"])
    assert tuple(encoded["input_ids"].shape) == (2, max_len)
    ids, mask = token_ids_numpy(texts, fake_tokenizer, cfg)
    assert ids.shape == (2, max_len)
    assert mask.shape == (2, max_len)

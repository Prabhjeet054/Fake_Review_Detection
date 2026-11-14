"""Tests for DistilBERT tokenization: CLS/SEP, padding, and truncation."""

from __future__ import annotations

import torch

from src.config import load_config
from src.tokenization import get_tokenizer, tokenize_texts

CLS_ID = 101
SEP_ID = 102
PAD_ID = 0


def test_tokenize_texts_shape_is_three_by_128() -> None:
    """Three sentences produce tensors of shape (3, 128)."""
    cfg = load_config()
    ids, mask = tokenize_texts(
        [
            "the hotel room was clean",
            "this blender is great quality",
            "the staff were helpful during our stay",
        ],
        cfg,
    )
    assert tuple(ids.shape) == (3, 128)
    assert tuple(mask.shape) == (3, 128)


def test_each_row_starts_with_cls_and_has_one_sep() -> None:
    """Every row starts with token 101 and contains exactly one 102."""
    cfg = load_config()
    ids, _mask = tokenize_texts(
        ["short review", "another hotel stay with friendly staff", "great product"],
        cfg,
    )
    for row in ids:
        assert int(row[0]) == CLS_ID
        assert int((row == SEP_ID).sum()) == 1


def test_padding_positions_are_zero_and_mask_sum_matches_tokens() -> None:
    """Where attention_mask is 0, input_ids is 0, and the mask sum is the real token count."""
    cfg = load_config()
    tokenizer = get_tokenizer(cfg)
    text = "the hotel was clean"
    ids, mask = tokenize_texts([text], cfg)
    assert torch.equal(ids[0][mask[0] == 0], torch.zeros_like(ids[0][mask[0] == 0]))
    real_count = len(
        tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=128)
    )
    assert int(mask[0].sum()) == real_count
    assert PAD_ID == 0


def test_thousand_word_text_is_truncated_and_ends_with_sep() -> None:
    """A 1,000-word text is truncated to 128 tokens and still ends with [SEP]."""
    cfg = load_config()
    text = " ".join(["hotel"] * 1000)
    ids, mask = tokenize_texts([text], cfg)
    assert tuple(ids.shape) == (1, 128)
    assert int(mask[0].sum()) == 128
    assert int(ids[0, -1]) == SEP_ID
    assert int(ids[0, 0]) == CLS_ID
    assert int((ids[0] == SEP_ID).sum()) == 1

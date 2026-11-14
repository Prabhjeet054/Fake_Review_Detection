"""Tests for ClassifierHead parameter counts, dropout, and DistilBertClassifier."""

from __future__ import annotations

import pytest
import torch

from src.config import load_config
from src.model import CLS_DIM, ClassifierHead, DistilBertClassifier, count_parameters

HEAD_TRAINABLE = 768 * 256 + 256 + 256 * 3 + 3  # 196,864 + 771 = 197,635
ENCODER_PARAMS = 66_362_880
TOTAL_PARAMS = 66_560_515


def test_classifier_head_has_197635_trainable_parameters() -> None:
    """ClassifierHead has exactly 197,635 trainable parameters."""
    assert HEAD_TRAINABLE == 197_635
    head = ClassifierHead(load_config())
    n_fc1 = count_parameters(head.fc1, trainable_only=True)
    n_fc2 = count_parameters(head.fc2, trainable_only=True)
    assert n_fc1 == 768 * 256 + 256
    assert n_fc2 == 256 * 3 + 3
    assert n_fc1 == 196_864
    assert n_fc2 == 771
    assert count_parameters(head, trainable_only=True) == HEAD_TRAINABLE


def test_classifier_head_output_shape_and_softmax() -> None:
    """A (4, 768) input yields (4, 3) logits whose softmax sums to 1 per row."""
    head = ClassifierHead(load_config())
    head.eval()
    logits = head(torch.randn(4, CLS_DIM))
    assert tuple(logits.shape) == (4, 3)
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)


def test_dropout_active_in_train_off_in_eval() -> None:
    """Two forwards differ in train() (dropout on) and match in eval()."""
    head = ClassifierHead(load_config())
    x = torch.randn(4, CLS_DIM)
    head.train()
    train_a = head(x)
    train_b = head(x)
    assert not torch.equal(train_a, train_b)
    head.eval()
    with torch.no_grad():
        eval_a = head(x)
        eval_b = head(x)
    assert torch.equal(eval_a, eval_b)


@pytest.mark.slow
def test_distilbert_classifier_parameter_counts() -> None:
    """Trainable head 197,635; frozen encoder 66,362,880; total 66,560,515."""
    model = DistilBertClassifier(load_config())
    trainable = count_parameters(model, trainable_only=True)
    encoder = count_parameters(model.encoder, trainable_only=False)
    total = count_parameters(model, trainable_only=False)
    assert trainable == HEAD_TRAINABLE
    assert encoder == ENCODER_PARAMS
    assert total == TOTAL_PARAMS
    assert trainable + encoder == total

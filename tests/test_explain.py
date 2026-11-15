"""Tests for attention and occlusion explanations."""

from types import SimpleNamespace

import pytest
import torch

from src.explain import (
    _last_layer_attention,
    _tokenize,
    attention_view,
    occlude_token,
    occlusion_view,
    render_html,
)


class FakeTokenizer:
    """Tokenizer fake exposing the interfaces used by attention_view."""

    all_special_tokens = {"[CLS]", "[SEP]", "[PAD]"}

    def __call__(self, texts, **kwargs):
        del texts, kwargs
        return {
            "input_ids": torch.tensor([[101, 10, 11, 12, 102, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]),
        }

    def convert_ids_to_tokens(self, token_ids):
        del token_ids
        return ["[CLS]", "great", "##ly", "product", "[SEP]", "[PAD]", "[PAD]", "[PAD]"]


class FakeEncoder:
    """Encoder fake returning one last-layer attention tensor."""

    def __call__(self, **kwargs):
        del kwargs
        attention = torch.zeros(1, 2, 8, 8)
        attention[:, :, :, :] = 1 / 8
        attention[:, :, 0, :] = 0.0
        attention[:, :, 0, 1] = 0.2
        attention[:, :, 0, 2] = 0.3
        attention[:, :, 0, 3] = 0.5
        attention[:, :, 0, 4:] = 0.0
        return SimpleNamespace(attentions=(attention,), last_hidden_state=torch.zeros(1, 8, 4))


class FakeClassifier:
    """Minimal classifier fake for explanation views."""

    def __init__(self):
        self.config = SimpleNamespace(model=SimpleNamespace(max_length=8))
        self.device = "cpu"
        self.tokenizer = FakeTokenizer()
        self.encoder = FakeEncoder()

    def predict(self, texts):
        """Return probabilities that decrease as words are removed."""
        probability = min(0.95, 0.5 + 0.05 * len(texts[0].split()))
        return [
            {
                "label": 0,
                "label_name": "Genuine Human",
                "probabilities": {
                    "Genuine Human": probability,
                    "Fake Human-Written": 1 - probability,
                    "AI-Generated": 0.0,
                },
                "cleaned_text": texts[0],
            }
        ]


def test_occlude_token_removes_one_word() -> None:
    """Occlusion removes exactly one whitespace token."""
    assert occlude_token(["best", "product", "ever"], 1) == "best ever"


def test_attention_merges_wordpieces_and_sums_weights() -> None:
    """Averaged attention rows normalize and WordPieces reconstruct words."""
    classifier = FakeClassifier()
    _, encoded = _tokenize(classifier, "greatly product")
    attention = _last_layer_attention(classifier, encoded)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(8))
    pairs = attention_view("greatly product", classifier)
    assert [word for word, _ in pairs] == ["greatly", "product"]
    assert [weight for _, weight in pairs] == pytest.approx([0.5, 0.5])
    assert " ".join(word for word, _ in pairs) == "greatly product"


def test_occlusion_returns_probability_drops() -> None:
    """Removing words returns one drop per original word."""
    pairs = occlusion_view("great product", FakeClassifier())
    assert [word for word, _ in pairs] == ["great", "product"]
    assert [drop for _, drop in pairs] == pytest.approx([0.05, 0.05])


def test_render_html_escapes_words_and_highlights_weights() -> None:
    """Rendered output is HTML-safe and contains weight metadata."""
    rendered = render_html([("<script>", 1.0), ("safe", -0.5)])
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "weight=1.0000" in rendered


def test_saved_explanations_exist_for_nine_examples() -> None:
    """The generated explanation directory contains all nine requested files."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    files = list((root / "outputs" / "figures" / "explanations").glob("*.html"))
    assert len(files) == 9

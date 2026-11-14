"""Tests for src.explain."""

from unittest.mock import patch

from src.explain import occlude_token, token_contributions
from src.model import build_classifier, save_classifier


def test_occlude_token() -> None:
    """Occlusion replaces a single token with the mask string."""
    assert occlude_token(["best", "product", "ever"], 1, "[MASK]") == "best [MASK] ever"


def test_token_contributions(cfg: dict, fake_tokenizer, fake_encoder) -> None:
    """Explanation returns one contribution per token for the predicted class."""
    save_classifier(build_classifier(cfg), cfg)
    with patch("src.explain.load_tokenizer", return_value=fake_tokenizer), patch(
        "src.explain.load_encoder", return_value=fake_encoder
    ):
        result = token_contributions("This product is amazing and great", cfg, device="cpu")
    assert result["label"] in {0, 1, 2}
    assert len(result["tokens"]) == len(result["contributions"])
    assert result["top_tokens"]

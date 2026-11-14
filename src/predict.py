"""Inference for a single review or a batch of texts."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.clean import clean_text
from src.config import label_names, load_config, parse_config_args
from src.embed import encode_texts, load_encoder
from src.model import load_classifier
from src.tokenization import load_tokenizer
from src.utils import get_device, seed_from_cfg, set_seed, setup_logging


def softmax_probs(logits: torch.Tensor) -> np.ndarray:
    """Convert logits to a numpy probability matrix."""
    return F.softmax(logits, dim=-1).detach().cpu().numpy()


def predict_texts(
    texts: list[str],
    cfg: dict[str, Any],
    *,
    tokenizer: Any | None = None,
    encoder: Any | None = None,
    classifier: Any | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Clean, embed, and classify raw review texts."""
    target = device or get_device(cfg)
    tokenizer = tokenizer or load_tokenizer(cfg)
    encoder = encoder or load_encoder(cfg, device=target)
    classifier = classifier or load_classifier(cfg, map_location=target)
    classifier.to(target)
    classifier.eval()
    cleaned = [clean_text(text, cfg) for text in texts]
    embeddings = encode_texts(cleaned, tokenizer, encoder, cfg, device=target)
    names = label_names(cfg)
    with torch.no_grad():
        tensor = torch.from_numpy(embeddings).to(target)
        logits = classifier(tensor)
        probs = softmax_probs(logits)
        pred_ids = probs.argmax(axis=1)
    results: list[dict[str, Any]] = []
    for text, cleaned_text, pred, prob_row in zip(texts, cleaned, pred_ids, probs):
        results.append(
            {
                "text": text,
                "text_clean": cleaned_text,
                "label": int(pred),
                "label_name": names[int(pred)],
                "probabilities": {names[i]: float(prob_row[i]) for i in range(len(prob_row))},
            }
        )
    return results


def main(argv: list[str] | None = None) -> None:
    """Classify a review passed with --text, or a short built-in demo sentence."""
    import argparse

    parser = argparse.ArgumentParser(description="Predict the class of a product review.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--text", default="This blender is fantastic and arrived quickly. I use it every morning.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    results = predict_texts([args.text], cfg)
    result = results[0]
    logger.info("Predicted %s (%s)", result["label_name"], result["label"])
    print(result["label_name"])
    for name, prob in result["probabilities"].items():
        print(f"{name}: {prob:.4f}")


if __name__ == "__main__":
    main()

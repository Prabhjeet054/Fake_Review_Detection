"""Token-occlusion explanations for a predicted review."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.clean import clean_text
from src.config import label_names, load_config, parse_config_args
from src.embed import encode_texts, load_encoder
from src.model import load_classifier
from src.predict import predict_texts
from src.tokenization import load_tokenizer
from src.utils import get_device, seed_from_cfg, set_seed, setup_logging


def occlude_token(tokens: list[str], index: int, mask_token: str) -> str:
    """Replace one whitespace token with the mask token."""
    copy = list(tokens)
    copy[index] = mask_token
    return " ".join(copy)


def token_contributions(
    text: str,
    cfg: dict[str, Any],
    *,
    tokenizer: Any | None = None,
    encoder: Any | None = None,
    classifier: Any | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Measure how each token changes the predicted-class probability when masked."""
    target = device or get_device(cfg)
    tokenizer = tokenizer or load_tokenizer(cfg)
    encoder = encoder or load_encoder(cfg, device=target)
    classifier = classifier or load_classifier(cfg, map_location=target)
    classifier.to(target)
    classifier.eval()
    cleaned = clean_text(text, cfg)
    tokens = cleaned.split()
    max_tokens = int(cfg["explain"]["max_tokens"])
    tokens = tokens[:max_tokens]
    if not tokens:
        return {"tokens": [], "contributions": [], "label": None, "label_name": None, "probability": 0.0}
    mask_token = str(cfg["explain"]["mask_token"])
    variants = [cleaned] + [occlude_token(tokens, i, mask_token) for i in range(len(tokens))]
    embeddings = encode_texts(variants, tokenizer, encoder, cfg, device=target)
    with torch.no_grad():
        logits = classifier(torch.from_numpy(embeddings).to(target))
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    pred = int(probs[0].argmax())
    base_prob = float(probs[0, pred])
    contribs = [base_prob - float(probs[i + 1, pred]) for i in range(len(tokens))]
    names = label_names(cfg)
    ranking = sorted(
        (
            {"token": token, "index": i, "contribution": float(score)}
            for i, (token, score) in enumerate(zip(tokens, contribs))
        ),
        key=lambda row: abs(row["contribution"]),
        reverse=True,
    )
    return {
        "text": text,
        "text_clean": cleaned,
        "label": pred,
        "label_name": names[pred],
        "probability": base_prob,
        "tokens": tokens,
        "contributions": contribs,
        "top_tokens": ranking[: int(cfg["explain"]["top_tokens"])],
    }


def main(argv: list[str] | None = None) -> None:
    """Print the top contributing tokens for a review."""
    import argparse

    parser = argparse.ArgumentParser(description="Explain a review prediction.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--text", default="This product is amazing. Best purchase ever. Highly recommend.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    prediction = predict_texts([args.text], cfg)[0]
    explanation = token_contributions(args.text, cfg)
    logger.info("Explanation for %s", explanation["label_name"])
    print(f"prediction={prediction['label_name']} p={explanation['probability']:.4f}")
    for row in explanation["top_tokens"]:
        print(f"{row['token']}\t{row['contribution']:.4f}")


if __name__ == "__main__":
    main()

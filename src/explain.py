"""Attention and occlusion explanations for review predictions."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.clean import clean_text
from src.config import AttrDict, load_config
from src.predict import ReviewClassifier


def _classifier(
    config: AttrDict | None = None,
    classifier: ReviewClassifier | None = None,
) -> ReviewClassifier:
    """Return the supplied classifier or load the configured one."""
    return classifier if classifier is not None else ReviewClassifier(config)


def _tokenize(classifier: ReviewClassifier, text: str) -> tuple[str, dict[str, torch.Tensor]]:
    """Clean and tokenize text with the production inference settings."""
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text must contain content after cleaning")
    encoded = classifier.tokenizer(
        [cleaned],
        padding="max_length",
        truncation=True,
        max_length=int(classifier.config.model.max_length),
        return_tensors="pt",
    )
    tensors = {key: value.to(classifier.device) for key, value in encoded.items()}
    return cleaned, tensors


def _merge_wordpieces(
    tokens: list[str],
    weights: list[float],
    special_tokens: set[str],
) -> list[tuple[str, float]]:
    """Merge ``##`` WordPiece tokens by summing their attention weights."""
    merged: list[list[Any]] = []
    for token, weight in zip(tokens, weights):
        if token in special_tokens or token.startswith("[PAD"):
            continue
        if token.startswith("##") and merged:
            merged[-1][0] += token[2:]
            merged[-1][1] += float(weight)
        else:
            merged.append([token, float(weight)])
    return [(str(word), float(weight)) for word, weight in merged]


def _last_layer_attention(
    classifier: ReviewClassifier,
    encoded: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return the last-layer attention matrix averaged over heads."""
    with torch.no_grad():
        output = classifier.encoder(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_attentions=True,
        )
    if not output.attentions:
        raise RuntimeError("The encoder did not return attention tensors.")
    return output.attentions[-1][0].mean(dim=0)


def attention_view(
    text: str,
    classifier: ReviewClassifier | None = None,
) -> list[tuple[str, float]]:
    """Return last-layer, head-averaged [CLS] attention merged into words."""
    model = _classifier(classifier=classifier)
    _, encoded = _tokenize(model, text)
    attention = _last_layer_attention(model, encoded)[0].detach().cpu().tolist()
    token_ids = encoded["input_ids"][0].detach().cpu().tolist()
    tokens = model.tokenizer.convert_ids_to_tokens(token_ids)
    special_tokens = set(model.tokenizer.all_special_tokens)
    return _merge_wordpieces(tokens, attention, special_tokens)


def occlude_token(tokens: list[str], index: int, mask_token: str = "") -> str:
    """Remove one whitespace token from a token list."""
    return " ".join(token for position, token in enumerate(tokens) if position != index)


def occlusion_view(
    text: str,
    classifier: ReviewClassifier | None = None,
) -> list[tuple[str, float]]:
    """Return each word's drop in predicted-class probability when removed."""
    model = _classifier(classifier=classifier)
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text must contain content after cleaning")
    words = cleaned.split()
    base = model.predict([cleaned])[0]
    predicted_name = str(base["label_name"])
    base_probability = float(base["probabilities"][predicted_name])
    pairs: list[tuple[str, float]] = []
    for index, word in enumerate(words):
        variant = occlude_token(words, index)
        if not variant:
            drop = base_probability
        else:
            probability = float(model.predict([variant])[0]["probabilities"][predicted_name])
            drop = base_probability - probability
        pairs.append((word, float(drop)))
    return pairs


def render_html(pairs: list[tuple[str, float]]) -> str:
    """Render words with color intensity proportional to absolute weight."""
    if not pairs:
        return "<p><em>No explanation available.</em></p>"
    maximum = max(abs(float(weight)) for _, weight in pairs) or 1.0
    spans = []
    for word, weight in pairs:
        intensity = min(abs(float(weight)) / maximum, 1.0)
        color = "rgba(220, 60, 60, " if weight >= 0 else "rgba(60, 100, 220, "
        spans.append(
            f'<span title="weight={float(weight):.4f}" '
            f'style="background-color:{color}{0.15 + 0.65 * intensity:.3f});'
            'padding:0.1rem 0.2rem;margin:0.1rem;display:inline-block;'
            f'border-radius:0.2rem">{html.escape(str(word))}</span>'
        )
    return " ".join(spans)


def _save_one_explanation(
    row: pd.Series,
    classifier: ReviewClassifier,
    output_dir: Path,
) -> Path:
    """Save both explanation views for one test review as an HTML file."""
    text = str(row["text"])
    attention = attention_view(text, classifier)
    occlusion = occlusion_view(text, classifier)
    predicted = classifier.predict([text])[0]
    file_name = f"id_{row['id']}_true_{int(row['true'])}_pred_{int(row['pred'])}.html"
    path = output_dir / file_name
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Review explanation {row['id']}</title>
<style>body{{font-family:system-ui,sans-serif;line-height:1.6;max-width:1000px;margin:2rem auto}}
.note{{color:#555}} .review{{background:#f5f5f5;padding:1rem;border-radius:.4rem}}</style>
</head><body>
<h1>Review explanation</h1>
<p><strong>True:</strong> {html.escape(str(row['true']))}
 &nbsp; <strong>Predicted:</strong> {html.escape(str(predicted['label_name']))}</p>
<p class="review">{html.escape(text)}</p>
<h2>Attention view</h2>
<p class="note">Attention comes from the frozen encoder and is not a causal explanation.</p>
<p>{render_html(attention)}</p>
<h2>Occlusion view</h2>
<p class="note">Occlusion directly measures the classifier's probability change.</p>
<p>{render_html(occlusion)}</p>
</body></html>
"""
    path.write_text(document, encoding="utf-8")
    return path


def save_test_explanations(config: AttrDict | None = None) -> list[Path]:
    """Save two correct and one incorrect explanation per true class."""
    cfg = config if config is not None else load_config()
    metrics_dir = Path(cfg.paths.metrics_dir)
    split_path = Path(cfg.paths.splits_dir) / "test.csv"
    predictions_path = metrics_dir / "test_predictions.csv"
    split = pd.read_csv(split_path, usecols=["id", "text"])
    predictions = pd.read_csv(predictions_path, usecols=["id", "true", "pred"])
    rows = predictions.merge(split, on="id", how="inner", validate="one_to_one")
    selected: list[pd.DataFrame] = []
    for label in range(int(cfg.model.num_classes)):
        selected.extend(
            [
                rows.loc[(rows["true"] == label) & (rows["pred"] == label)].head(2),
                rows.loc[(rows["true"] == label) & (rows["pred"] != label)].head(1),
            ]
        )
    chosen = pd.concat(selected, ignore_index=True)
    output_dir = Path(cfg.paths.figures_dir) / "explanations"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ReviewClassifier(cfg)
    return [_save_one_explanation(row, model, output_dir) for _, row in chosen.iterrows()]


def token_contributions(
    text: str,
    cfg: AttrDict,
    *,
    tokenizer: Any | None = None,
    encoder: Any | None = None,
    classifier: Any | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper exposing occlusion scores in the legacy shape."""
    model = ReviewClassifier(
        cfg,
        tokenizer=tokenizer,
        encoder=encoder,
        head=classifier,
        device=device,
    )
    pairs = occlusion_view(text, model)
    result = model.predict([text])[0]
    ranking = sorted(
        (
            {"token": word, "index": index, "contribution": float(drop)}
            for index, (word, drop) in enumerate(pairs)
        ),
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )
    return {
        "text": text,
        "text_clean": result["cleaned_text"],
        "label": result["label"],
        "label_name": result["label_name"],
        "probability": result["probabilities"][result["label_name"]],
        "tokens": [word for word, _ in pairs],
        "contributions": [drop for _, drop in pairs],
        "top_tokens": ranking[: int(getattr(cfg, "explain", {}).get("top_tokens", 5))],
    }


def main(argv: list[str] | None = None) -> None:
    """Save explanations for selected correctly and incorrectly classified tests."""
    parser = argparse.ArgumentParser(description="Save test-review explanations.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args(argv)
    paths = save_test_explanations(load_config(args.config))
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

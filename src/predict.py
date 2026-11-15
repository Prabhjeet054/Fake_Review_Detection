"""Single-review and batch inference for the trained fake-review classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.clean import clean_text
from src.config import AttrDict, load_config
from src.embed import load_encoder
from src.model import ClassifierHead
from src.tokenization import get_tokenizer
from src.utils import get_device, set_seed


def softmax_probs(logits: torch.Tensor) -> np.ndarray:
    """Convert logits to a NumPy probability matrix."""
    return F.softmax(logits, dim=-1).detach().cpu().numpy()


def _label_names(config: AttrDict) -> dict[int, str]:
    """Return configured class names keyed by integer label."""
    return {int(key): str(value) for key, value in config.labels.items()}


def _load_head(config: AttrDict, device: str) -> nn.Module:
    """Load the saved best classification head onto the selected device."""
    path = Path(config.paths.models_dir) / "head_best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run python -m src.train first.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    head = ClassifierHead(config)
    state_dict = payload["state_dict"] if isinstance(payload, dict) else payload
    head.load_state_dict(state_dict)
    head.to(device)
    head.eval()
    return head


class ReviewClassifier:
    """Load and reuse the tokenizer, frozen encoder, and trained head."""

    def __init__(
        self,
        config: AttrDict | None = None,
        *,
        device: str | None = None,
        tokenizer: Any | None = None,
        encoder: nn.Module | None = None,
        head: nn.Module | None = None,
    ) -> None:
        """Load inference components once, allowing dependency injection in tests."""
        self.config = config if config is not None else load_config()
        self.device = device or get_device()
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer(self.config)
        if encoder is None:
            self.encoder, encoder_device = load_encoder(self.config)
            self.device = device or encoder_device
        else:
            self.encoder = encoder
        self.encoder.to(self.device)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.head = head if head is not None else _load_head(self.config, self.device)
        self.head.to(self.device)
        self.head.eval()
        self.names = _label_names(self.config)

    def _clean_texts(self, texts: list[str]) -> list[str]:
        """Clean texts and reject empty input or empty cleaned reviews."""
        if not texts:
            raise ValueError("texts must not be empty")
        cleaned = [clean_text(text) for text in texts]
        if any(not value for value in cleaned):
            raise ValueError("Every text must contain content after cleaning")
        return cleaned

    def _tokenize(self, cleaned: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize using the same padding, truncation, and max length as training."""
        encoded = self.tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=int(self.config.model.max_length),
            return_tensors="pt",
        )
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Return labels, probabilities, and cleaned text for each review."""
        cleaned = self._clean_texts(texts)
        input_ids, attention_mask = self._tokenize(cleaned)
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        embeddings = hidden[:, 0, :]
        probabilities = softmax_probs(self.head(embeddings))
        labels = probabilities.argmax(axis=1)
        return [
            {
                "label": int(label),
                "label_name": self.names[int(label)],
                "probabilities": {
                    self.names[index]: float(probability)
                    for index, probability in enumerate(row)
                },
                "cleaned_text": cleaned_text,
            }
            for label, row, cleaned_text in zip(labels, probabilities, cleaned)
        ]


def predict_texts(
    texts: list[str],
    config: AttrDict,
    *,
    tokenizer: Any | None = None,
    encoder: nn.Module | None = None,
    head: nn.Module | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Compatibility wrapper around :class:`ReviewClassifier`."""
    return ReviewClassifier(
        config,
        device=device,
        tokenizer=tokenizer,
        encoder=encoder,
        head=head,
    ).predict(texts)


def _write_batch_predictions(
    texts: list[str],
    results: list[dict[str, Any]],
    path: Path,
) -> Path:
    """Write batch predictions with one probability column per class."""
    rows = []
    for text, result in zip(texts, results):
        row = {
            "text": text,
            "cleaned_text": result["cleaned_text"],
            "label": result["label"],
            "label_name": result["label_name"],
        }
        row.update(result["probabilities"])
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    """Classify one review or a CSV file containing a ``text`` column."""
    parser = argparse.ArgumentParser(description="Classify fake-review text.")
    parser.add_argument("text", nargs="?", help="Review text to classify.")
    parser.add_argument("--file", type=Path, help="CSV file containing a text column.")
    parser.add_argument("--out", type=Path, help="Output CSV path for --file predictions.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    args = parser.parse_args(argv)
    has_text = args.text is not None
    has_file = args.file is not None
    if has_text == has_file:
        parser.error("Provide exactly one review text or --file.")
    if args.file is not None and args.out is None:
        parser.error("--out is required with --file.")

    config = load_config(args.config)
    set_seed(int(config.seed))
    classifier = ReviewClassifier(config)
    if args.file is not None:
        frame = pd.read_csv(args.file)
        if "text" not in frame.columns:
            raise ValueError(f"{args.file} must contain a 'text' column")
        texts = frame["text"].fillna("").astype(str).tolist()
        results = classifier.predict(texts)
        output = _write_batch_predictions(texts, results, args.out)
        print(f"wrote {output}")
        return

    result = classifier.predict([args.text])[0]
    print(f"{result['label_name']} (class {result['label']})")
    for name, probability in result["probabilities"].items():
        print(f"{name}: {probability:.4f}")


if __name__ == "__main__":
    main()

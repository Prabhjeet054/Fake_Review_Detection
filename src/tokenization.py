"""DistilBERT tokenization helpers. Named tokenization.py so it does not shadow stdlib tokenize."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import load_config, parse_config_args, path_from_cfg
from src.features import texts_and_labels
from src.utils import load_dataframe, seed_from_cfg, set_seed, setup_logging


def load_tokenizer(cfg: dict[str, Any]) -> Any:
    """Load the Hugging Face tokenizer named in config."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg["model"]["encoder"])


def tokenize_texts(
    texts: list[str],
    tokenizer: Any,
    cfg: dict[str, Any],
    *,
    return_tensors: str | None = "pt",
) -> dict[str, Any]:
    """Tokenize texts with padding/truncation to the configured max length."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(cfg["model"]["max_length"]),
        return_tensors=return_tensors,
    )


def token_ids_numpy(texts: list[str], tokenizer: Any, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return input_ids and attention_mask as numpy arrays."""
    encoded = tokenize_texts(texts, tokenizer, cfg, return_tensors=None)
    input_ids = np.array(encoded["input_ids"], dtype=np.int64)
    mask = np.array(encoded["attention_mask"], dtype=np.int64)
    return input_ids, mask


def main(argv: list[str] | None = None) -> None:
    """Tokenize the train split and print tensor shapes."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    train = load_dataframe(path_from_cfg(cfg, "paths.splits") / "train.csv")
    texts, _ = texts_and_labels(train, cfg)
    tokenizer = load_tokenizer(cfg)
    sample = texts[: min(4, len(texts))]
    encoded = tokenize_texts(sample, tokenizer, cfg)
    shape = tuple(encoded["input_ids"].shape)
    logger.info("Tokenized sample shape %s (max_length=%s)", shape, cfg["model"]["max_length"])
    print(f"input_ids.shape={shape}")


if __name__ == "__main__":
    main()

"""Compute and cache frozen DistilBERT [CLS] embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, parse_config_args, path_from_cfg
from src.features import texts_and_labels
from src.tokenization import load_tokenizer, tokenize_texts
from src.utils import ensure_dir, get_device, load_dataframe, save_json, seed_from_cfg, set_seed, setup_logging


def load_encoder(cfg: dict[str, Any], device: str | None = None) -> Any:
    """Load DistilBERT in eval mode with frozen weights."""
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(cfg["model"]["encoder"])
    if cfg["model"]["freeze_encoder"]:
        for param in model.parameters():
            param.requires_grad = False
    target = device or get_device(cfg)
    model.to(target)
    model.eval()
    return model


def cls_from_hidden(last_hidden_state: Any) -> Any:
    """Select the [CLS] token embedding from the last hidden state."""
    return last_hidden_state[:, 0, :]


def encode_texts(
    texts: list[str],
    tokenizer: Any,
    encoder: Any,
    cfg: dict[str, Any],
    device: str | None = None,
) -> np.ndarray:
    """Embed texts in batches and return a float32 array of shape (n, embedding_dim)."""
    import torch

    target = device or get_device(cfg)
    batch_size = int(cfg["embed"]["batch_size"])
    dim = int(cfg["model"]["embedding_dim"])
    outputs = np.zeros((len(texts), dim), dtype=np.float32)
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenize_texts(batch, tokenizer, cfg, return_tensors="pt")
            encoded = {key: value.to(target) for key, value in encoded.items()}
            hidden = encoder(**encoded).last_hidden_state
            cls_vec = cls_from_hidden(hidden).detach().cpu().numpy().astype(np.float32)
            outputs[start : start + len(batch)] = cls_vec
    return outputs


def split_embedding_paths(cfg: dict[str, Any], split: str) -> dict[str, Path]:
    """Return npy/csv/json paths for a named split."""
    directory = ensure_dir(path_from_cfg(cfg, "paths.embeddings"))
    return {
        "embeddings": directory / f"{split}_cls.npy",
        "labels": directory / f"{split}_labels.npy",
        "ids": directory / f"{split}_ids.csv",
        "meta": directory / f"{split}_meta.json",
    }


def save_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ids: list[str],
    cfg: dict[str, Any],
    split: str,
) -> dict[str, Path]:
    """Persist embeddings, labels, ids, and metadata for one split."""
    paths = split_embedding_paths(cfg, split)
    np.save(paths["embeddings"], embeddings.astype(np.float32))
    np.save(paths["labels"], labels.astype(np.int64))
    pd.DataFrame({cfg["columns"]["id"]: ids}).to_csv(paths["ids"], index=False)
    save_json(
        {
            "split": split,
            "encoder": cfg["model"]["encoder"],
            "max_length": cfg["model"]["max_length"],
            "embedding_dim": int(embeddings.shape[1]),
            "n": int(embeddings.shape[0]),
            "freeze_encoder": cfg["model"]["freeze_encoder"],
        },
        paths["meta"],
    )
    return paths


def load_embeddings(cfg: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load cached embeddings for a split."""
    paths = split_embedding_paths(cfg, split)
    if not paths["embeddings"].exists():
        raise FileNotFoundError(f"Missing embeddings for split '{split}': {paths['embeddings']}")
    embeddings = np.load(paths["embeddings"])
    labels = np.load(paths["labels"])
    ids = pd.read_csv(paths["ids"])[cfg["columns"]["id"]].astype(str).tolist()
    return embeddings, labels, ids


def embed_split(
    df: pd.DataFrame,
    tokenizer: Any,
    encoder: Any,
    cfg: dict[str, Any],
    split: str,
    device: str | None = None,
) -> dict[str, Path]:
    """Embed one split DataFrame and cache the arrays."""
    texts, labels = texts_and_labels(df, cfg)
    ids = df[cfg["columns"]["id"]].astype(str).tolist()
    vectors = encode_texts(texts, tokenizer, encoder, cfg, device=device)
    return save_embeddings(vectors, labels, ids, cfg, split)


def main(argv: list[str] | None = None) -> None:
    """Compute cached [CLS] embeddings for train/val/test."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    device = get_device(cfg)
    tokenizer = load_tokenizer(cfg)
    encoder = load_encoder(cfg, device=device)
    split_dir = path_from_cfg(cfg, "paths.splits")
    for split in ("train", "val", "test"):
        df = load_dataframe(split_dir / f"{split}.csv")
        paths = embed_split(df, tokenizer, encoder, cfg, split, device=device)
        logger.info("Cached %s embeddings n=%s -> %s", split, len(df), paths["embeddings"])
        print(f"{split}: {paths['embeddings']}")


if __name__ == "__main__":
    main()

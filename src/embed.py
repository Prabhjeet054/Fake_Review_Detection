"""Compute and cache frozen DistilBERT [CLS] embeddings per split."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel

from src.config import AttrDict, load_config
from src.tokenization import tokenize_texts
from src.utils import ensure_dirs, get_device, get_logger, set_seed

SPLIT_NAMES = ("train", "val", "test")
CLS_DIM = 768


def load_encoder(config: AttrDict) -> tuple[torch.nn.Module, str]:
    """Load AutoModel, freeze every parameter, eval(), and move it to get_device()."""
    device = get_device()
    model = AutoModel.from_pretrained(str(config.model.name))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    return model, device


def cls_from_hidden(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Return the [CLS] vector last_hidden_state[:, 0, :]."""
    return last_hidden_state[:, 0, :]


def embedding_paths(config: AttrDict, split: str) -> dict[str, Path]:
    """Return npy paths for one split's X, y, and ids."""
    directory = Path(config.paths.embeddings_dir)
    return {
        "X": directory / f"{split}_X.npy",
        "y": directory / f"{split}_y.npy",
        "ids": directory / f"{split}_ids.npy",
    }


def split_files_exist(config: AttrDict, split: str) -> bool:
    """Return True if X, y, and ids npy files already exist for the split."""
    return all(path.is_file() for path in embedding_paths(config, split).values())


def embed_texts(
    texts: list[str],
    config: AttrDict,
    encoder: torch.nn.Module,
    device: str,
    desc: str,
) -> np.ndarray:
    """Embed texts in batches and return float32 [CLS] vectors of shape (n, 768)."""
    batch_size = int(config.training.embed_batch_size)
    chunks: list[np.ndarray] = []
    starts = range(0, len(texts), batch_size)
    for start in tqdm(starts, desc=desc, total=(len(texts) + batch_size - 1) // batch_size):
        batch = texts[start : start + batch_size]
        input_ids, attention_mask = tokenize_texts(batch, config)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            hidden = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            cls_vec = cls_from_hidden(hidden).detach().cpu().numpy().astype(np.float32)
        chunks.append(cls_vec)
    if not chunks:
        return np.zeros((0, CLS_DIM), dtype=np.float32)
    return np.vstack(chunks)


def embed_split(
    config: AttrDict,
    split: str,
    encoder: torch.nn.Module,
    device: str,
    force: bool = False,
) -> dict[str, Path]:
    """Embed one split, skipping if cached unless force is True."""
    logger = get_logger(__name__)
    paths = embedding_paths(config, split)
    if split_files_exist(config, split) and not force:
        logger.info("Skipping %s embeddings; files already exist. Pass --force to recompute.", split)
        print(f"skip {split}: {paths['X']}")
        return paths
    csv_path = Path(config.paths.splits_dir) / f"{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing split file: {csv_path}. Run python -m src.split first.")
    df = pd.read_csv(csv_path)
    texts = df["text_clean"].fillna("").astype(str).tolist()
    started = time.perf_counter()
    X = embed_texts(texts, config, encoder, device, desc=f"embed {split}")
    elapsed = time.perf_counter() - started
    y = df["label"].astype(int).to_numpy()
    ids = df["id"].to_numpy()
    Path(config.paths.embeddings_dir).mkdir(parents=True, exist_ok=True)
    np.save(paths["X"], X.astype(np.float32))
    np.save(paths["y"], y)
    np.save(paths["ids"], ids)
    logger.info("Embedded %s n=%s dim=%s in %.2fs -> %s", split, X.shape[0], X.shape[1], elapsed, paths["X"])
    print(f"{split}: X={tuple(X.shape)} dtype={X.dtype} time={elapsed:.2f}s saved={paths['X']}")
    return paths


def embed_all_splits(config: AttrDict | None = None, force: bool = False) -> None:
    """Embed train, val, and test splits with a frozen DistilBERT encoder."""
    cfg = config if config is not None else load_config()
    set_seed(int(cfg.seed))
    ensure_dirs(cfg)
    encoder, device = load_encoder(cfg)
    print(f"device={device} model={cfg.model.name}")
    for split in SPLIT_NAMES:
        embed_split(cfg, split, encoder, device, force=force)


def main(argv: list[str] | None = None) -> None:
    """CLI: embed all splits, skipping existing files unless --force."""
    parser = argparse.ArgumentParser(description="Cache frozen DistilBERT [CLS] embeddings.")
    parser.add_argument("--force", action="store_true", help="Recompute even if npy files exist.")
    args = parser.parse_args(argv)
    embed_all_splits(load_config(), force=bool(args.force))


if __name__ == "__main__":
    main()

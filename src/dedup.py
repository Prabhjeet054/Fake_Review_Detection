"""Near-duplicate grouping so similar reviews stay in the same split."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from src.config import load_config, parse_config_args, path_from_cfg
from src.utils import load_dataframe, save_dataframe, seed_from_cfg, set_seed, setup_logging


class UnionFind:
    """Disjoint-set structure for grouping near-duplicate reviews."""

    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in self.parent}

    def find(self, item: str) -> str:
        """Return the representative for an item, with path compression."""
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        """Merge the sets containing left and right."""
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def shingles(text: str, ngram: int, level: str) -> set[str]:
    """Return character or word n-grams for similarity comparison."""
    cleaned = (text or "").strip()
    if not cleaned:
        return set()
    if level == "word":
        tokens = cleaned.split()
        if len(tokens) < ngram:
            return {" ".join(tokens)} if tokens else set()
        return {" ".join(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)}
    if len(cleaned) < ngram:
        return {cleaned}
    return {cleaned[i : i + ngram] for i in range(len(cleaned) - ngram + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    """Compute Jaccard similarity of two shingle sets."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _group_exact(df: pd.DataFrame, text_col: str, id_col: str) -> UnionFind:
    """Union reviews that share an exact cleaned-text hash."""
    uf = UnionFind(df[id_col].astype(str).tolist())
    buckets: dict[str, str] = {}
    for review_id, text in zip(df[id_col].astype(str), df[text_col].fillna("").astype(str)):
        key = " ".join(text.split())
        if key in buckets:
            uf.union(buckets[key], review_id)
        else:
            buckets[key] = review_id
    return uf


def _group_bruteforce(
    df: pd.DataFrame,
    uf: UnionFind,
    cfg: dict[str, Any],
) -> UnionFind:
    """Pairwise Jaccard grouping for small collections."""
    settings = cfg["dedup"]
    text_col = cfg["columns"]["clean_text"]
    id_col = cfg["columns"]["id"]
    ids = df[id_col].astype(str).tolist()
    gram_sets = [
        shingles(text, int(settings["ngram"]), str(settings["ngram_level"]))
        for text in df[text_col].fillna("").astype(str)
    ]
    threshold = float(settings["threshold"])
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if jaccard(gram_sets[i], gram_sets[j]) >= threshold:
                uf.union(ids[i], ids[j])
    return uf


def _group_minhash(
    df: pd.DataFrame,
    uf: UnionFind,
    cfg: dict[str, Any],
) -> UnionFind:
    """Approximate near-duplicate grouping with MinHash LSH."""
    from datasketch import MinHash, MinHashLSH

    settings = cfg["dedup"]
    text_col = cfg["columns"]["clean_text"]
    id_col = cfg["columns"]["id"]
    num_perm = int(settings["num_perm"])
    threshold = float(settings["threshold"])
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    signatures: dict[str, MinHash] = {}
    for review_id, text in zip(df[id_col].astype(str), df[text_col].fillna("").astype(str)):
        grams = shingles(text, int(settings["ngram"]), str(settings["ngram_level"]))
        mh = MinHash(num_perm=num_perm)
        if not grams:
            mh.update(b"")
        for gram in grams:
            mh.update(gram.encode("utf-8"))
        signatures[review_id] = mh
        neighbours = lsh.query(mh)
        for neighbour in neighbours:
            uf.union(review_id, neighbour)
        lsh.insert(review_id, mh)
    return uf


def assign_group_ids(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add a group_id column for exact and near-duplicate reviews."""
    if cfg["columns"]["clean_text"] not in df.columns:
        raise ValueError("Cleaned text column is required before near-duplicate grouping.")
    id_col = cfg["columns"]["id"]
    group_col = cfg["columns"]["group"]
    uf = _group_exact(df, cfg["columns"]["clean_text"], id_col)
    limit = int(cfg["dedup"]["brute_force_limit"])
    method = str(cfg["dedup"]["method"]).lower()
    if len(df) <= limit or method == "jaccard":
        uf = _group_bruteforce(df, uf, cfg)
    else:
        uf = _group_minhash(df, uf, cfg)
    out = df.copy()
    out[group_col] = out[id_col].astype(str).map(lambda review_id: f"g_{uf.find(review_id)}")
    return out


def main(argv: list[str] | None = None) -> None:
    """Group near-duplicates in the cleaned interim file."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    df = load_dataframe(path_from_cfg(cfg, "data.interim_clean"))
    grouped = assign_group_ids(df, cfg)
    out_path = path_from_cfg(cfg, "data.interim_grouped")
    save_dataframe(grouped, out_path)
    n_groups = grouped[cfg["columns"]["group"]].nunique()
    logger.info("Grouped %s reviews into %s groups -> %s", len(grouped), n_groups, out_path)
    print(f"wrote {out_path} rows={len(grouped)} groups={n_groups}")


if __name__ == "__main__":
    main()

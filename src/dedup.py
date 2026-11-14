"""Drop exact duplicate reviews and group near-duplicates for a leak-free split."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.config import AttrDict, load_config
from src.utils import ensure_dirs, get_logger, set_seed


class UnionFind:
    """Disjoint-set structure that merges near-duplicate row indices."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, item: int) -> int:
        """Return the representative for an item, with path compression."""
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
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


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first row for each text_clean value and log drops per label."""
    logger = get_logger(__name__)
    duplicate = df.duplicated(subset=["text_clean"], keep="first")
    n_dropped = int(duplicate.sum())
    if n_dropped and "label" in df.columns:
        counts = df.loc[duplicate, "label"].value_counts().sort_index()
        for label, count in counts.items():
            logger.info("Dropped %s exact duplicate rows (label=%s)", int(count), int(label))
    else:
        logger.info("Dropped %s exact duplicate rows", n_dropped)
    return df.loc[~duplicate].reset_index(drop=True)


def _similarity_text(text: str) -> str:
    """Map punctuation to spaces so short punctuation-only edits still count as near-dups."""
    mapped = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in str(text))
    return re.sub(r"\s+", " ", mapped).strip()


def link_near_duplicates(texts: list[str], threshold: float) -> UnionFind:
    """Union rows whose char 3-5 gram TF-IDF cosine similarity is at least threshold."""
    uf = UnionFind(len(texts))
    if len(texts) <= 1:
        return uf
    comparable = [_similarity_text(text) for text in texts]
    matrix = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1).fit_transform(comparable)
    radius = max(0.0, 1.0 - float(threshold))
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
    neighbors.fit(matrix)
    distances, indices = neighbors.radius_neighbors(matrix, radius=radius, return_distance=True)
    for i, (dist_row, index_row) in enumerate(zip(distances, indices)):
        for dist, j in zip(dist_row, index_row):
            if int(j) > i and float(dist) <= radius:
                uf.union(i, int(j))
    return uf


def assign_group_ids(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add a group_id for every row; singletons receive their own id."""
    texts = df["text_clean"].fillna("").astype(str).tolist()
    uf = link_near_duplicates(texts, threshold)
    root_to_group: dict[int, int] = {}
    group_ids: list[int] = []
    next_id = 0
    for index in range(len(df)):
        root = uf.find(index)
        if root not in root_to_group:
            root_to_group[root] = next_id
            next_id += 1
        group_ids.append(root_to_group[root])
    out = df.copy()
    out["group_id"] = group_ids
    return out


def _print_example_groups(df: pd.DataFrame, sizes: pd.Series, n_examples: int = 3) -> None:
    """Print up to n_examples multi-member groups with truncated cleaned text."""
    multi = sizes[sizes > 1].sort_values(ascending=False)
    if multi.empty:
        print("No near-duplicate groups with more than one member.")
        return
    for group_id in multi.head(n_examples).index:
        group = df.loc[df["group_id"] == group_id]
        print(f"\n--- group_id={int(group_id)} n={len(group)} ---")
        for _, row in group.iterrows():
            preview = str(row["text_clean"]).replace("\n", " ")[:180]
            print(f"[{row.get('id', '')} label={row.get('label', '')}] {preview}")


def run_dedup(df: pd.DataFrame, config: AttrDict | None = None) -> pd.DataFrame:
    """Drop exact duplicates, group near-duplicates, and save dataset.csv."""
    cfg = config if config is not None else load_config()
    logger = get_logger(__name__)
    set_seed(int(cfg.seed))
    out = drop_exact_duplicates(df)
    out = assign_group_ids(out, float(cfg.dedup.near_dup_threshold))
    sizes = out.groupby("group_id").size()
    n_multi = int((sizes > 1).sum())
    logger.info("%s groups have more than one member", n_multi)
    print(f"multi_member_groups={n_multi}")
    _print_example_groups(out, sizes)
    ensure_dirs(cfg)
    out_path = Path(cfg.paths.processed_dir) / "dataset.csv"
    out.to_csv(out_path, index=False)
    print(f"saved={out_path} n_rows={len(out)} n_groups={out['group_id'].nunique()}")
    return out


def main() -> None:
    """Deduplicate data/interim/cleaned.csv and write data/processed/dataset.csv."""
    cfg = load_config()
    path = Path(cfg.paths.interim_dir) / "cleaned.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {path}. Run python -m src.clean first."
        )
    run_dedup(pd.read_csv(path), cfg)


if __name__ == "__main__":
    main()

"""Load Salminen and Ott corpora into the canonical review schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config, parse_config_args, path_from_cfg
from src.schema import add_label_names, empty_reviews_frame, validate_reviews
from src.utils import save_dataframe, seed_from_cfg, set_seed, setup_logging


def _canonical_frame(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> pd.DataFrame:
    """Build a canonical DataFrame from row dicts."""
    if not rows:
        return empty_reviews_frame(cfg)
    df = pd.DataFrame(rows)
    return add_label_names(df, cfg)


def load_salminen(cfg: dict[str, Any], csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the Salminen fake-reviews CSV and map OR/CG labels."""
    path = Path(csv_path) if csv_path is not None else path_from_cfg(cfg, "paths.salminen_csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Salminen dataset not found at {path}. "
            "Download from https://osf.io/tyue9/ and place it at the configured path."
        )
    raw = pd.read_csv(path)
    sal_cfg = cfg["data"]["salminen"]
    text_col = sal_cfg["text_column"]
    label_col = sal_cfg["label_column"]
    if text_col not in raw.columns:
        fallback = cfg["columns"]["text"]
        if fallback in raw.columns:
            text_col = fallback
        else:
            raise ValueError(f"Salminen CSV missing text column '{sal_cfg['text_column']}'.")
    if label_col not in raw.columns:
        raise ValueError(f"Salminen CSV missing label column '{label_col}'.")
    mapping = {str(key).upper(): int(value) for key, value in cfg["label_map"]["salminen"].items()}
    columns = cfg["columns"]
    rows: list[dict[str, Any]] = []
    for index, row in raw.iterrows():
        raw_label = str(row[label_col]).strip().upper()
        if raw_label not in mapping:
            raise ValueError(f"Unknown Salminen label '{raw_label}' at row {index}.")
        category = row[sal_cfg["category_column"]] if sal_cfg["category_column"] in raw.columns else None
        rating = row[sal_cfg["rating_column"]] if sal_cfg["rating_column"] in raw.columns else None
        rows.append(
            {
                columns["id"]: f"{sal_cfg['source_name']}_{int(index):06d}",
                columns["text"]: str(row[text_col]),
                columns["label"]: mapping[raw_label],
                columns["source"]: sal_cfg["source_name"],
                columns["category"]: None if pd.isna(category) else str(category),
                columns["rating"]: None if pd.isna(rating) else float(rating),
                columns["polarity"]: None,
                columns["hotel"]: None,
            }
        )
    return validate_reviews(_canonical_frame(rows, cfg), cfg)


def _relative_parts(path: Path, root: Path | None) -> list[str]:
    """Return lowercased path parts relative to the corpus root when possible."""
    resolved = path.resolve()
    if root is not None:
        try:
            resolved = resolved.relative_to(Path(root).resolve())
        except ValueError:
            pass
    return [part.lower() for part in resolved.parts]


def _ott_label_from_path(path: Path, cfg: dict[str, Any], root: Path | None = None) -> int:
    """Infer truthful/deceptive label from folder names under the Ott root."""
    parts = _relative_parts(path, root)
    mapping = {str(key).lower(): int(value) for key, value in cfg["label_map"]["ott"].items()}
    for key, label in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        if any(key in part for part in parts):
            return label
    raise ValueError(f"Could not infer Ott label from path: {path}")


def _ott_polarity_from_path(path: Path, root: Path | None = None) -> str | None:
    """Infer review polarity from folder names under the Ott root."""
    parts = _relative_parts(path, root)
    if any("positive" in part for part in parts):
        return "positive"
    if any("negative" in part for part in parts):
        return "negative"
    return None


def _ott_hotel_from_path(path: Path) -> str | None:
    """Best-effort hotel name from the Ott filename stem."""
    stem = path.stem
    pieces = stem.split("_")
    if len(pieces) >= 2:
        return pieces[1]
    return stem or None


def load_ott(cfg: dict[str, Any], ott_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the Ott deceptive-opinion corpus from text files."""
    root = Path(ott_dir) if ott_dir is not None else path_from_cfg(cfg, "paths.ott_dir")
    if not root.exists():
        raise FileNotFoundError(
            f"Ott corpus not found at {root}. "
            "Download the Deceptive Opinion Spam Corpus and extract it to the configured path."
        )
    pattern = cfg["data"]["ott"]["glob"]
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No Ott .txt files found under {root} with glob '{pattern}'.")
    columns = cfg["columns"]
    source_name = cfg["data"]["ott"]["source_name"]
    rows: list[dict[str, Any]] = []
    for index, file_path in enumerate(files):
        text = file_path.read_text(encoding="utf-8", errors="replace")
        rows.append(
            {
                columns["id"]: f"{source_name}_{index:06d}",
                columns["text"]: text,
                columns["label"]: _ott_label_from_path(file_path, cfg, root=root),
                columns["source"]: source_name,
                columns["category"]: "hotel",
                columns["rating"]: None,
                columns["polarity"]: _ott_polarity_from_path(file_path, root=root),
                columns["hotel"]: _ott_hotel_from_path(file_path),
            }
        )
    return validate_reviews(_canonical_frame(rows, cfg), cfg)


def combine_datasets(frames: list[pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    """Concatenate loaded datasets and re-validate the schema."""
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        raise ValueError("No reviews loaded from any source.")
    combined = pd.concat(nonempty, ignore_index=True)
    return validate_reviews(combined, cfg)


def load_all(cfg: dict[str, Any], *, allow_missing: bool = False) -> pd.DataFrame:
    """Load every available configured source and concatenate them."""
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    try:
        frames.append(load_salminen(cfg))
    except FileNotFoundError as exc:
        errors.append(str(exc))
        if not allow_missing:
            raise
    try:
        frames.append(load_ott(cfg))
    except FileNotFoundError as exc:
        errors.append(str(exc))
        if not allow_missing:
            raise
    if not frames:
        raise FileNotFoundError(" ; ".join(errors))
    return combine_datasets(frames, cfg)


def main(argv: list[str] | None = None) -> None:
    """Load raw datasets and write the interim combined CSV."""
    args = parse_config_args(argv)
    cfg = load_config(args.config)
    set_seed(seed_from_cfg(cfg))
    logger = setup_logging(cfg)
    df = load_all(cfg, allow_missing=False)
    out_path = path_from_cfg(cfg, "data.interim_raw")
    save_dataframe(df, out_path)
    logger.info("Loaded %s reviews -> %s", len(df), out_path)
    print(df[cfg["columns"]["label"]].value_counts().sort_index().to_string())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""Tests for src.dedup using a 5-row exact/near/unrelated fixture."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config
from src.dedup import run_dedup

TEXT_A = "this phone is great battery lasts long"
TEXT_A_NEAR = "this phone is great, battery lasts long!"
TEXT_D = "terrible hotel, the room was dirty"
TEXT_E = "the pasta was cold and bland"


def _five_row_fixture() -> pd.DataFrame:
    """Return five rows: A, an exact copy of A, a near copy of A, D, and E."""
    return pd.DataFrame(
        {
            "id": ["A", "A_copy", "A_near", "D", "E"],
            "label": [0, 0, 0, 1, 2],
            "source": ["salminen"] * 5,
            "domain": ["product", "product", "product", "hotel", "product"],
            "text_clean": [TEXT_A, TEXT_A, TEXT_A_NEAR, TEXT_D, TEXT_E],
        }
    )


def test_five_row_fixture_drops_exact_and_groups_near_copy(tmp_path: Path) -> None:
    """Exact copy of A is dropped; A shares group_id with its near copy; D and E are unique."""
    cfg = load_config()
    cfg.paths.processed_dir = tmp_path
    out = run_dedup(_five_row_fixture(), cfg)
    assert len(out) == 4
    assert "A_copy" not in set(out["id"])
    by_id = out.set_index("id")
    assert by_id.loc["A", "group_id"] == by_id.loc["A_near", "group_id"]
    assert by_id.loc["D", "group_id"] != by_id.loc["A", "group_id"]
    assert by_id.loc["E", "group_id"] != by_id.loc["A", "group_id"]
    assert by_id.loc["D", "group_id"] != by_id.loc["E", "group_id"]
    assert out["group_id"].notna().all()
    assert out["group_id"].nunique() == 3
    assert set(out["group_id"]) == set(by_id["group_id"])
    assert (tmp_path / "dataset.csv").is_file()

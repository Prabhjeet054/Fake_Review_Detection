"""In-memory DataFrame checks for src.schema."""

from __future__ import annotations

import pandas as pd
import pytest

from src.schema import ID2LABEL, LABEL2ID, validate_dataframe


def _valid_frame() -> pd.DataFrame:
    """Return a 3-row frame with one example of each class."""
    return pd.DataFrame(
        {
            "id": ["r0", "r1", "r2"],
            "text": ["This blender works well.", "The hotel stay was fabricated.", "Best product ever generated."],
            "label": [0, 1, 2],
            "source": ["salminen", "ott", "salminen"],
            "domain": ["product", "hotel", "product"],
        }
    )


def test_valid_dataframe_returned_unchanged() -> None:
    """A valid 3-row DataFrame passes and is returned unchanged."""
    df = _valid_frame()
    out = validate_dataframe(df)
    assert out is df
    assert len(out) == 3


def test_missing_domain_column_raises() -> None:
    """A missing domain column raises ValueError mentioning domain."""
    df = _valid_frame().drop(columns=["domain"])
    with pytest.raises(ValueError, match="domain"):
        validate_dataframe(df)


def test_label_three_raises() -> None:
    """A label of 3 raises ValueError."""
    df = _valid_frame()
    df.loc[0, "label"] = 3
    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_blank_and_nan_text_raise() -> None:
    """A text of '   ' and a NaN text each raise ValueError."""
    blank = _valid_frame()
    blank.loc[0, "text"] = "   "
    with pytest.raises(ValueError):
        validate_dataframe(blank)
    nan_text = _valid_frame()
    nan_text.loc[1, "text"] = float("nan")
    with pytest.raises(ValueError):
        validate_dataframe(nan_text)


def test_duplicate_ids_raise() -> None:
    """Duplicate ids raise ValueError."""
    df = _valid_frame()
    df.loc[2, "id"] = "r0"
    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_label_maps_round_trip() -> None:
    """ID2LABEL[1] is Fake Human-Written and LABEL2ID round-trips every class."""
    assert ID2LABEL[1] == "Fake Human-Written"
    for label_id, name in ID2LABEL.items():
        assert LABEL2ID[name] == label_id
        assert ID2LABEL[LABEL2ID[name]] == name

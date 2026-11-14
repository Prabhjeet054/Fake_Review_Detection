"""Tests for run_all.py."""

import pytest

from run_all import STEP_ORDER, select_steps


def test_select_all_steps_by_default() -> None:
    """With no filters, every pipeline step is selected."""
    assert select_steps(None, None) == STEP_ORDER


def test_from_step_and_subset() -> None:
    """from-step slices the pipeline; unknown names raise."""
    steps = select_steps(None, "train")
    assert steps[0] == "train"
    assert steps[-1] == "summary"
    assert select_steps(["clean", "split"], None) == ["clean", "split"]
    with pytest.raises(ValueError):
        select_steps(["not-a-step"], None)

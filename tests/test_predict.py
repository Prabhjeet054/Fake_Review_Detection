"""Integration tests for inference against the saved test predictions."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.predict import ReviewClassifier

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "outputs" / "metrics"


@pytest.fixture(scope="module")
def classifier() -> ReviewClassifier:
    """Load the production classifier once for this test module."""
    return ReviewClassifier(load_config())


@pytest.fixture(scope="module")
def test_predictions() -> pd.DataFrame:
    """Load saved predictions produced by the evaluation pipeline."""
    return pd.read_csv(METRICS_DIR / "test_predictions.csv")


@pytest.fixture(scope="module")
def test_rows() -> pd.DataFrame:
    """Load test rows containing the original, uncleaned review text."""
    return pd.read_csv(ROOT / "data" / "processed" / "splits" / "test.csv")


def test_raw_text_matches_saved_predictions(
    classifier: ReviewClassifier,
    test_predictions: pd.DataFrame,
    test_rows: pd.DataFrame,
) -> None:
    """Raw-text inference matches saved labels and probabilities within 1e-4."""
    rng = np.random.default_rng(42)
    sample = test_predictions.iloc[rng.choice(len(test_predictions), size=20, replace=False)]
    raw_by_id = test_rows.set_index("id")["text"].to_dict()
    texts = [str(raw_by_id[row_id]) for row_id in sample["id"]]
    actual = classifier.predict(texts)

    expected_labels = sample["pred"].to_numpy(dtype=int)
    expected_probabilities = sample[["p0", "p1", "p2"]].to_numpy(dtype=float)
    actual_labels = np.array([result["label"] for result in actual])
    actual_probabilities = np.array(
        [
            [
                result["probabilities"]["Genuine Human"],
                result["probabilities"]["Fake Human-Written"],
                result["probabilities"]["AI-Generated"],
            ]
            for result in actual
        ]
    )
    assert np.array_equal(actual_labels, expected_labels)
    assert np.allclose(actual_probabilities, expected_probabilities, atol=1e-4)


@pytest.mark.parametrize("text", ["", "★★★"])
def test_empty_or_symbol_only_review_is_rejected(
    classifier: ReviewClassifier, text: str
) -> None:
    """Reviews with no usable cleaned text raise ValueError."""
    with pytest.raises(ValueError):
        classifier.predict([text])


def test_two_thousand_word_review_runs(classifier: ReviewClassifier) -> None:
    """Long reviews are truncated by the configured tokenizer without failing."""
    result = classifier.predict(["excellent product " * 1000])
    assert len(result) == 1


def test_probabilities_sum_to_one(
    classifier: ReviewClassifier, test_predictions: pd.DataFrame
) -> None:
    """Every predicted probability distribution sums to one."""
    results = classifier.predict(test_predictions["text_clean"].head(20).tolist())
    for result in results:
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_average_single_review_latency_ms(
    classifier: ReviewClassifier, test_rows: pd.DataFrame
) -> None:
    """Time 100 single-review predictions and print average latency."""
    text = str(test_rows.iloc[0]["text"])
    started = time.perf_counter()
    for _ in range(100):
        classifier.predict([text])
    average_ms = (time.perf_counter() - started) * 1000 / 100
    print(f"average single-review latency: {average_ms:.2f} ms")
    assert average_ms > 0

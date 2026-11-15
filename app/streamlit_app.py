"""Streamlit interface for the fake and AI-generated review detector."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AttrDict, load_config
from src.explain import attention_view, occlusion_view, render_html
from src.predict import ReviewClassifier

CLASS_NAMES = ("Genuine Human", "Fake Human-Written", "AI-Generated")
BADGE_COLORS = {
    "Genuine Human": "#2e7d32",
    "Fake Human-Written": "#ef6c00",
    "AI-Generated": "#c62828",
}


@st.cache_resource
def load_classifier(config_path: str) -> ReviewClassifier:
    """Load and cache the inference classifier for the Streamlit session."""
    return ReviewClassifier(load_config(config_path))


@st.cache_data
def load_configured_data(config_path: str) -> tuple[AttrDict, pd.DataFrame]:
    """Load config and correctly classified test examples."""
    cfg = load_config(config_path)
    predictions_path = Path(cfg.paths.metrics_dir) / "test_predictions.csv"
    split_path = Path(cfg.paths.splits_dir) / "test.csv"
    predictions = pd.read_csv(predictions_path)
    split = pd.read_csv(split_path, usecols=["id", "text"])
    correct = predictions.loc[predictions["true"] == predictions["pred"], ["id", "true"]]
    examples = correct.merge(split, on="id", how="inner")
    return cfg, examples


def _metrics(cfg: AttrDict) -> dict[str, Any]:
    """Read saved test metrics."""
    path = Path(cfg.paths.metrics_dir) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _prediction_frame(texts: list[str], results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a flat DataFrame from classifier results."""
    rows: list[dict[str, Any]] = []
    for text, result in zip(texts, results):
        row = {
            "text": text,
            "cleaned_text": result["cleaned_text"],
            "label": result["label"],
            "label_name": result["label_name"],
        }
        row.update(result["probabilities"])
        rows.append(row)
    return pd.DataFrame(rows)


def _show_prediction(result: dict[str, Any]) -> None:
    """Render a prediction badge, probability chart, and cleaned text."""
    label_name = str(result["label_name"])
    color = BADGE_COLORS[label_name]
    st.markdown(
        f'<div style="background:{color};color:white;padding:0.7rem 1rem;'
        f'border-radius:0.5rem;font-size:1.2rem;font-weight:600;">'
        f"Predicted class: {label_name}</div>",
        unsafe_allow_html=True,
    )
    probabilities = pd.DataFrame(
        {"Probability": list(result["probabilities"].values())},
        index=list(result["probabilities"].keys()),
    )
    st.bar_chart(probabilities, horizontal=True)
    with st.expander("Cleaned text"):
        st.write(result["cleaned_text"])


def _show_explanation(text: str, classifier: ReviewClassifier) -> None:
    """Render attention and occlusion explanations for a prediction."""
    with st.expander("Explain this prediction"):
        st.caption(
            "Attention comes from the frozen encoder and is not a causal explanation. "
            "Occlusion directly measures the classifier's probability change."
        )
        with st.spinner("Computing explanation..."):
            attention = attention_view(text, classifier)
            occlusion = occlusion_view(text, classifier)
        st.markdown("**Attention view**")
        st.markdown(render_html(attention), unsafe_allow_html=True)
        st.markdown("**Occlusion view**")
        st.markdown(render_html(occlusion), unsafe_allow_html=True)


def _render_sidebar(cfg: AttrDict) -> None:
    """Render model details and saved test metrics."""
    metrics = _metrics(cfg)
    st.sidebar.header("Model details")
    st.sidebar.write(f"**Encoder:** `{cfg.model.name}`")
    st.sidebar.write("**Head:** Linear(768, 256) → ReLU → Dropout(0.3) → Linear(256, 3)")
    st.sidebar.write(f"**Epochs:** {cfg.training.epochs}")
    st.sidebar.metric("Test accuracy", f"{float(metrics.get('accuracy', 0.0)):.4f}")
    st.sidebar.metric("Test macro F1", f"{float(metrics.get('macro_f1', 0.0)):.4f}")


def _render_examples(examples: pd.DataFrame) -> None:
    """Render one correctly classified example button for each class."""
    st.subheader("Try an example")
    columns = st.columns(3)
    for index, (column, class_name) in enumerate(zip(columns, CLASS_NAMES)):
        matches = examples.loc[examples["true"] == index]
        if matches.empty:
            column.info(f"No {class_name} example available.")
            continue
        text = str(matches.iloc[0]["text"])
        if column.button(class_name, key=f"example_{index}"):
            st.session_state["review_text"] = text


def _render_single_tab(cfg: AttrDict, examples: pd.DataFrame, config_path: str) -> None:
    """Render the single-review analysis tab."""
    _render_examples(examples)
    review = st.text_area(
        "Review text",
        height=180,
        key="review_text",
        placeholder="Paste a product review here...",
    )
    if not st.button("Analyze", type="primary"):
        return
    if not review.strip():
        st.warning("Enter a review before analyzing.")
        return
    try:
        classifier = load_classifier(config_path)
        result = classifier.predict([review])[0]
    except ValueError as exc:
        st.warning(str(exc))
        return
    _show_prediction(result)
    _show_explanation(review, classifier)


def _render_batch_tab(config_path: str) -> None:
    """Render CSV upload, batch results, and download controls."""
    uploaded = st.file_uploader("Upload a CSV with a text column", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to classify multiple reviews.")
        return
    frame = pd.read_csv(uploaded)
    if "text" not in frame.columns:
        st.warning("The uploaded CSV must contain a 'text' column.")
        return
    texts = frame["text"].fillna("").astype(str).tolist()
    try:
        results = load_classifier(config_path).predict(texts)
    except ValueError as exc:
        st.warning(str(exc))
        return
    output = _prediction_frame(texts, results)
    st.dataframe(output, use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions CSV",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    """Render the review detector application."""
    st.set_page_config(page_title="Fake & AI-Generated Review Detector", layout="wide")
    config_path = str(ROOT / "config.yaml")
    cfg, examples = load_configured_data(config_path)
    st.title("Fake & AI-Generated Review Detector")
    st.caption("Classify product reviews as genuine, fake human-written, or AI-generated.")
    _render_sidebar(cfg)
    analyze_tab, batch_tab = st.tabs(["Analyze", "Batch"])
    with analyze_tab:
        _render_single_tab(cfg, examples, config_path)
    with batch_tab:
        _render_batch_tab(config_path)


if __name__ == "__main__":
    main()

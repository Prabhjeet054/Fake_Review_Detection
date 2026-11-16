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
    "Genuine Human": "#0b6e4f",
    "Fake Human-Written": "#b45309",
    "AI-Generated": "#6d28d9",
}
CLASS_HINTS = (
    "A real customer wrote this.",
    "A person wrote this to deceive.",
    "A model generated this text.",
)
STYLES_PATH = Path(__file__).resolve().parent / "styles.css"


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


def _inject_styles() -> None:
    """Load the local stylesheet into the Streamlit page."""
    if not STYLES_PATH.is_file():
        return
    st.markdown(f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _metrics(cfg: AttrDict) -> dict[str, Any]:
    """Read saved test metrics."""
    path = Path(cfg.paths.metrics_dir) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _confidence(result: dict[str, Any]) -> float:
    """Return the predicted-class probability."""
    name = str(result["label_name"])
    return float(result["probabilities"].get(name, 0.0))


def _prediction_frame(texts: list[str], results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a flat DataFrame from classifier results."""
    rows: list[dict[str, Any]] = []
    for text, result in zip(texts, results):
        row = {
            "text": text,
            "cleaned_text": result["cleaned_text"],
            "label": result["label"],
            "label_name": result["label_name"],
            "confidence": _confidence(result),
        }
        row.update(result["probabilities"])
        rows.append(row)
    return pd.DataFrame(rows)


def _show_prediction(result: dict[str, Any]) -> None:
    """Render a prediction badge, probability bars, and cleaned text."""
    label_name = str(result["label_name"])
    color = BADGE_COLORS[label_name]
    confidence = _confidence(result)
    st.markdown(
        f'<div class="pred-badge" style="background:{color};">'
        f'<span class="pred-kicker">Predicted class</span>'
        f'<span class="pred-name">{label_name}</span>'
        f'<span class="pred-conf">{confidence:.0%} confidence</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption("Class probabilities")
    for name, probability in result["probabilities"].items():
        st.text(f"{name} · {float(probability):.1%}")
        st.progress(min(max(float(probability), 0.0), 1.0))
    with st.expander("Cleaned text used by the model"):
        st.write(result["cleaned_text"])


def _show_explanation(text: str, classifier: ReviewClassifier) -> None:
    """Render attention and occlusion explanations for a prediction."""
    with st.expander("Why this score?", expanded=True):
        st.markdown(
            '<p class="explain-note">Attention is the frozen encoder looking at tokens; '
            "it is not a causal explanation. Occlusion measures how the predicted "
            "probability changes when a word is removed.</p>",
            unsafe_allow_html=True,
        )
        with st.spinner("Computing token highlights..."):
            attention = attention_view(text, classifier)
            occlusion = occlusion_view(text, classifier)
        left, right = st.columns(2)
        with left:
            st.markdown("**Attention**")
            st.markdown(render_html(attention), unsafe_allow_html=True)
        with right:
            st.markdown("**Occlusion**")
            st.markdown(render_html(occlusion), unsafe_allow_html=True)


def _render_sidebar(cfg: AttrDict) -> None:
    """Render model details and saved test metrics."""
    metrics = _metrics(cfg)
    st.sidebar.markdown("### Detector")
    st.sidebar.caption("Frozen DistilBERT CLS token plus a small MLP head. Seed 42.")
    st.sidebar.metric("Test accuracy", f"{float(metrics.get('accuracy', 0.0)):.1%}")
    st.sidebar.metric("Macro F1", f"{float(metrics.get('macro_f1', 0.0)):.1%}")
    st.sidebar.divider()
    st.sidebar.markdown("**Encoder**")
    st.sidebar.code(str(cfg.model.name), language=None)
    st.sidebar.markdown("**Head**")
    st.sidebar.caption("Linear(768 → 256) → ReLU → Dropout(0.3) → Linear(256 → 3)")
    st.sidebar.markdown("**Training**")
    st.sidebar.caption(f"{int(cfg.training.epochs)} epochs · max length {int(cfg.model.max_length)}")


def _render_examples(examples: pd.DataFrame) -> None:
    """Render one correctly classified example button for each class."""
    st.markdown("**Start from a labeled example**")
    st.caption("Each button loads a held-out review the model already got right.")
    columns = st.columns(3)
    for index, (column, class_name, hint) in enumerate(zip(columns, CLASS_NAMES, CLASS_HINTS)):
        matches = examples.loc[examples["true"] == index]
        if matches.empty:
            column.info(f"No {class_name} example available.")
            continue
        text = str(matches.iloc[0]["text"])
        with column:
            if st.button(class_name, key=f"example_{index}", use_container_width=True):
                st.session_state["review_text"] = text
            st.caption(hint)


def _render_single_tab(cfg: AttrDict, examples: pd.DataFrame, config_path: str) -> None:
    """Render the single-review analysis tab."""
    del cfg
    left, right = st.columns((1.05, 0.95), gap="large")
    with left, st.container(border=True):
        _render_examples(examples)
        review = st.text_area(
            "Review text",
            height=200,
            key="review_text",
            placeholder="Paste a hotel or product review…",
        )
        analyze = st.button("Analyze", type="primary", use_container_width=True)
    with right:
        if not analyze:
            st.markdown(
                '<div class="empty-panel"><strong>Waiting for a review</strong>'
                "<p>Load an example or paste text, then analyze. "
                "The predicted class, probabilities, and token highlights appear here.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            return
        if not review.strip():
            st.warning("Enter a review before analyzing.")
            return
        try:
            classifier = load_classifier(config_path)
            with st.spinner("Scoring review…"):
                result = classifier.predict([review])[0]
        except ValueError as exc:
            st.warning(str(exc))
            return
        with st.container(border=True):
            _show_prediction(result)
            _show_explanation(review, classifier)


def _render_batch_tab(config_path: str) -> None:
    """Render CSV upload, batch results, and download controls."""
    st.markdown("**Score many reviews at once**")
    st.caption("The CSV must include a text column. Other columns are ignored.")
    with st.container(border=True):
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
        with st.spinner(f"Scoring {len(texts)} reviews…"):
            results = load_classifier(config_path).predict(texts)
    except ValueError as exc:
        st.warning(str(exc))
        return
    output = _prediction_frame(texts, results)
    counts = output["label_name"].value_counts()
    metrics = st.columns(3)
    for column, name in zip(metrics, CLASS_NAMES):
        column.metric(name, int(counts.get(name, 0)))
    st.dataframe(
        output,
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1),
            "text": st.column_config.TextColumn("text", width="large"),
        },
    )
    st.download_button(
        "Download predictions CSV",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    """Render the review detector application."""
    st.set_page_config(
        page_title="Fake & AI-Generated Review Detector",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    config_path = str(ROOT / "config.yaml")
    cfg, examples = load_configured_data(config_path)
    st.markdown('<p class="hero-kicker">Review authenticity</p>', unsafe_allow_html=True)
    st.title("Fake & AI-Generated Review Detector")
    st.markdown(
        '<p class="hero-lede">Paste a hotel or product review. A frozen DistilBERT encoder '
        "and a small classification head score whether it reads as genuine, "
        "human-written deception, or machine-generated text.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="legend">'
        '<span><i class="g"></i> Genuine</span>'
        '<span><i class="f"></i> Human-written fake</span>'
        '<span><i class="a"></i> Machine-generated</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    _render_sidebar(cfg)
    analyze_tab, batch_tab = st.tabs(["Analyze", "Batch"])
    with analyze_tab:
        _render_single_tab(cfg, examples, config_path)
    with batch_tab:
        _render_batch_tab(config_path)
    st.markdown(
        '<p class="footer-note">Head-only training on frozen embeddings · seed 42 · '
        "not a production moderation system</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

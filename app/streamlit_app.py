"""Streamlit demo for classifying a product review."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, path_from_cfg
from src.model import classifier_path
from src.predict import predict_texts
from src.utils import set_seed, seed_from_cfg


def get_config() -> dict:
    """Load config.yaml from the project root."""
    return load_config(ROOT / "config.yaml")


@st.cache_resource
def load_bundle():
    """Load tokenizer, encoder, and classifier once per session."""
    from src.embed import load_encoder
    from src.model import load_classifier
    from src.tokenization import load_tokenizer
    from src.utils import get_device

    cfg = get_config()
    set_seed(seed_from_cfg(cfg))
    device = get_device(cfg)
    tokenizer = load_tokenizer(cfg)
    encoder = load_encoder(cfg, device=device)
    classifier = load_classifier(cfg, map_location=device)
    return cfg, tokenizer, encoder, classifier, device


def main() -> None:
    """Render the Streamlit classification UI."""
    cfg = get_config()
    st.set_page_config(page_title="Fake Review Detector", layout="centered")
    st.title("Fake Review Detector")
    st.caption(
        "Classify a product review as Genuine Human, Fake Human-Written, or AI-Generated. "
        "Settings are read from config.yaml."
    )
    model_file = classifier_path(cfg)
    if not model_file.exists():
        st.warning(
            "No trained classifier found. Run the pipeline first:\n\n"
            "`PYTHONHASHSEED=42 python run_all.py`"
        )
        st.stop()

    cfg, tokenizer, encoder, classifier, device = load_bundle()
    review = st.text_area("Review text", height=180, placeholder="Paste a product review here...")
    explain = st.checkbox("Show token-occlusion explanation", value=False)
    if st.button("Classify", type="primary") and review.strip():
        results = predict_texts(
            [review],
            cfg,
            tokenizer=tokenizer,
            encoder=encoder,
            classifier=classifier,
            device=device,
        )
        result = results[0]
        st.subheader(result["label_name"])
        probs = pd.DataFrame(
            {"class": list(result["probabilities"].keys()), "probability": list(result["probabilities"].values())}
        )
        st.bar_chart(probs.set_index("class"))
        st.json(result["probabilities"])
        if explain:
            from src.explain import token_contributions

            with st.spinner("Computing token contributions..."):
                explanation = token_contributions(
                    review,
                    cfg,
                    tokenizer=tokenizer,
                    encoder=encoder,
                    classifier=classifier,
                    device=device,
                )
            top = pd.DataFrame(explanation["top_tokens"])
            if not top.empty:
                st.subheader("Top contributing tokens")
                st.dataframe(top, hide_index=True)

    metrics_md = path_from_cfg(cfg, "paths.metrics") / "results_summary.md"
    if metrics_md.exists():
        with st.expander("Latest experiment results"):
            st.markdown(metrics_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

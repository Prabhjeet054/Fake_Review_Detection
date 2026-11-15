"""Tests for app/streamlit_app.py."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def test_app_renders_detector_tabs() -> None:
    """The demo renders the detector title and both analysis modes."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    assert app.title[0].value == "Fake & AI-Generated Review Detector"
    assert len(app.tabs) == 2


def test_empty_review_shows_warning() -> None:
    """Clicking Analyze with no text shows a warning instead of crashing."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    analyze = next(button for button in app.button if button.label == "Analyze")
    analyze.click().run()
    assert any("Enter a review" in str(block.value) for block in app.warning)

"""AppTest coverage for the Streamlit review detector."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
pytestmark = pytest.mark.slow
CLASS_NAMES = ("Genuine Human", "Fake Human-Written", "AI-Generated")


def _app() -> AppTest:
    """Create an AppTest instance for the Streamlit app."""
    return AppTest.from_file(str(APP_PATH), default_timeout=60)


def _run_analyze(app: AppTest, text: str) -> AppTest:
    """Set review text, click Analyze, and rerun the app."""
    app.text_area[0].set_value(text)
    analyze = next(button for button in app.button if button.label == "Analyze")
    analyze.click()
    return app.run()


def test_app_loads_without_exceptions() -> None:
    """The app starts without uncaught exceptions."""
    app = _app().run()
    assert not app.exception


def test_analyzing_review_shows_one_class_name() -> None:
    """Analyzing a review renders exactly one configured class name."""
    app = _run_analyze(_app().run(), "This product works well and arrived quickly.")
    assert not app.exception
    rendered = "\n".join(str(block.value) for block in app.markdown)
    assert sum(name in rendered for name in CLASS_NAMES) == 1


def test_empty_analyze_shows_warning_without_exception() -> None:
    """Clicking Analyze with empty input shows a warning and no exception."""
    app = _run_analyze(_app().run(), "")
    assert not app.exception
    assert any("Enter a review" in str(block.value) for block in app.warning)


def test_each_example_button_fills_text_area() -> None:
    """Each class example button fills the review text area."""
    app = _app().run()
    for index in range(3):
        app.button(f"example_{index}").click()
        app.run()
        assert app.text_area[0].value.strip()

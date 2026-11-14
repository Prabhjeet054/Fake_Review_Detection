"""Tests for app/streamlit_app.py."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def test_app_prompts_when_untrained() -> None:
    """The demo warns the user until a classifier checkpoint exists."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    assert app.title[0].value == "Fake Review Detector"
    assert any("No trained classifier" in str(block.value) for block in app.warning)

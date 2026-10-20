"""Runs the Streamlit demo in-process. Skipped where torch/transformers are not installed (CI)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("torch")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "model" / "final" / "config.json").exists():
    pytest.skip("model/final not available", allow_module_level=True)

from streamlit.testing.v1 import AppTest  # noqa: E402


def test_demo_loads_validates_input_and_rate_limits():
    at = AppTest.from_file(str(ROOT / "streamlit_demo" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception
    assert at.text_area[0].max_chars == 5000          # the size limit is enforced by the widget

    at.button[0].click().run()                         # empty input
    assert "Please enter some findings." in [e.value for e in at.error]

    at.text_area[0].set_value("The lungs are clear. No focal consolidation.")
    at.button[0].click().run()
    assert at.success and not at.error                 # a real summary comes back

    for _ in range(5):                                 # per-session limit is 5 a minute
        at.button[0].click().run()
    assert any("Too many requests" in e.value for e in at.error)

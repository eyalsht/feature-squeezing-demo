"""Tests for app.py UI helpers.

Because app.py runs _build_services() at module scope (loading insightface),
we patch the heavy service constructors before importing the module.
"""
from __future__ import annotations

import sys
import importlib
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to import app with mocked services
# ---------------------------------------------------------------------------

def _import_app():
    """Import (or return cached) app module with insightface mocked out."""
    if "app" in sys.modules:
        return sys.modules["app"]

    mock_embedder = MagicMock()
    v = np.random.randn(512).astype(np.float32)
    mock_embedder.embed.return_value = v / np.linalg.norm(v)

    mock_detector = MagicMock()
    mock_squeeze_detector = MagicMock()
    mock_db = MagicMock()
    mock_db.names.return_value = ["alice"]

    with (
        patch("detector.FaceDetector", return_value=mock_detector),
        patch("detector.ArcFaceEmbedder", return_value=mock_embedder),
        patch("detector.SqueezeDetector", return_value=mock_squeeze_detector),
        patch("dataset.IdentityDatabase", return_value=mock_db),
    ):
        import app as _app
    return _app


# ---------------------------------------------------------------------------
# Commit 2: multi-threshold bar chart
# ---------------------------------------------------------------------------

def test_threshold_lines_at_paper_and_demo_values():
    """Bar-chart figure must have axvlines at x≈0.72 (paper) and x≈0.50 (demo)."""
    app = _import_app()
    sims = {"original": 0.9, "attacked": 0.85, "bit": 0.4, "median": 0.35, "nlm": 0.3}
    fig = app._render_similarity_bars(sims)
    ax = fig.axes[0]
    xvals = [line.get_xdata()[0] for line in ax.get_lines()]
    assert any(abs(x - 0.72) < 0.001 for x in xvals), (
        f"Expected axvline at x≈0.72 but got xvals={xvals}"
    )
    assert any(abs(x - 0.50) < 0.001 for x in xvals), (
        f"Expected axvline at x≈0.50 but got xvals={xvals}"
    )
    assert len([x for x in xvals if abs(x - 0.72) < 0.001 or abs(x - 0.50) < 0.001]) >= 2


# ---------------------------------------------------------------------------
# Commit 5: registration validation
# ---------------------------------------------------------------------------

def test_register_raises_on_missing_inputs():
    """on_register must raise gr.Error when name or photos are missing."""
    import gradio as gr
    app = _import_app()
    with pytest.raises(gr.Error):
        app.on_register("", None, None, None)
    with pytest.raises(gr.Error):
        app.on_register("Alice", None, None, None)
    with pytest.raises(gr.Error):
        img = np.zeros((160, 160, 3), dtype=np.uint8)
        app.on_register("", img, img, None)

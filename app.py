"""Feature Squeezing Live Demo — Gradio entry point.

Hugging Face Spaces requires ``app.py`` at the repository root. All business
logic lives in ``src/`` modules, so we add ``src/`` to ``sys.path`` before
importing project modules. Tests resolve the same paths via ``pytest.ini``
(`pythonpath = src`).
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── sys.path bootstrap (MUST run before importing project modules) ───────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

# LOCAL-ONLY: patch gradio_client schema walker to tolerate bool `additionalProperties`.
# REVERT BEFORE HF DEPLOY: delete this block — HF Spaces uses a compatible gradio_client.
import gradio_client.utils as _gc_utils  # noqa: E402

_orig_get_type = _gc_utils.get_type
_orig_json_to_py = _gc_utils._json_schema_to_python_type


def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _orig_get_type(schema)


def _safe_json_to_py(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_to_py(schema, defs)


_gc_utils.get_type = _safe_get_type
_gc_utils._json_schema_to_python_type = _safe_json_to_py

import numpy as np
import gradio as gr
import matplotlib

# LOCAL-ONLY: also short-circuit Blocks.get_api_info — gradio still calls it on
# every page request and the bool-schema bug surfaces there too.
# REVERT BEFORE HF DEPLOY: delete this block.
_EMPTY_API_INFO = {"named_endpoints": {}, "unnamed_endpoints": {}}
gr.Blocks.get_api_info = lambda self: _EMPTY_API_INFO

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detector import FaceDetector, ArcFaceEmbedder, SqueezeDetector
from squeezers import BitDepthSqueezer, MedianFilterSqueezer, NonLocalMeansSqueezer
from dataset import IdentityDatabase
from attack import Face, GlassesAttacker


# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
  --bg-base:     #0a0a14;
  --bg-surface:  #14142a;
  --bg-card:     rgba(255,255,255,0.03);
  --border-dim:  rgba(255,255,255,0.08);
  --border-glow: rgba(167,139,250,0.35);
  --purple:      #a78bfa;
  --purple-dim:  rgba(167,139,250,0.15);
  --red:         #ef4444;
  --red-dim:     rgba(239,68,68,0.12);
  --green:       #10b981;
  --green-dim:   rgba(16,185,129,0.12);
  --amber:       #fbbf24;
  --text-muted:  #64748b;
  --text-body:   #94a3b8;
  --text-bright: #e2e8f0;
}

body, .gradio-container {
  background:
    radial-gradient(ellipse at top left, rgba(167,139,250,0.10), transparent 50%),
    radial-gradient(ellipse at bottom right, rgba(239,68,68,0.06), transparent 55%),
    var(--bg-base) !important;
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  color: var(--text-bright) !important;
}

.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  padding: 16px !important;
}

/* Panels & boxes */
.gr-panel, .gr-box, .gr-block, .gr-group {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: 12px !important;
}

/* Labels: ALL CAPS micro-labels */
label, .gr-label, .label-wrap span {
  color: var(--text-muted) !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 1.4px !important;
}

/* Buttons — smooth hover with transform + glow */
.gr-button, button.lg {
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  letter-spacing: 0.5px !important;
  padding: 10px 16px !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
  border: 1px solid var(--border-dim) !important;
}
.gr-button:hover, button.lg:hover {
  transform: translateY(-1px) !important;
}

.btn-attack {
  background: linear-gradient(135deg, rgba(239,68,68,0.20), rgba(239,68,68,0.08)) !important;
  border: 1px solid var(--red) !important;
  color: #fca5a5 !important;
}
.btn-attack:hover {
  box-shadow: 0 0 24px rgba(239,68,68,0.45) !important;
  background: linear-gradient(135deg, rgba(239,68,68,0.30), rgba(239,68,68,0.14)) !important;
}

.btn-defend {
  background: linear-gradient(135deg, rgba(16,185,129,0.20), rgba(16,185,129,0.08)) !important;
  border: 1px solid var(--green) !important;
  color: #6ee7b7 !important;
}
.btn-defend:hover {
  box-shadow: 0 0 24px rgba(16,185,129,0.45) !important;
  background: linear-gradient(135deg, rgba(16,185,129,0.30), rgba(16,185,129,0.14)) !important;
}

.btn-register {
  background: linear-gradient(135deg, rgba(167,139,250,0.18), rgba(167,139,250,0.06)) !important;
  border: 1px solid var(--purple) !important;
  color: #c4b5fd !important;
}
.btn-register:hover {
  box-shadow: 0 0 22px rgba(167,139,250,0.45) !important;
}

/* Squeezer hero card — glassmorphism */
.squeezer-card {
  background: linear-gradient(135deg,
              rgba(167,139,250,0.10),
              rgba(167,139,250,0.02)) !important;
  border: 1px solid var(--border-glow) !important;
  border-radius: 16px !important;
  padding: 20px 24px !important;
  backdrop-filter: blur(14px) saturate(140%) !important;
  -webkit-backdrop-filter: blur(14px) saturate(140%) !important;
  box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04) !important;
  margin-bottom: 16px !important;
}

/* Slider styling */
input[type=range] {
  accent-color: var(--purple) !important;
}
.gr-slider .min, .gr-slider .max {
  color: var(--text-muted) !important;
}
/* larger numeric readouts */
.gr-slider input[type=number],
.gr-number input {
  font-family: 'Inter', sans-serif !important;
  font-size: 18px !important;
  font-weight: 700 !important;
  color: var(--purple) !important;
  background: rgba(167,139,250,0.06) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: 8px !important;
}

/* Tabs */
.tab-nav button {
  color: var(--text-muted) !important;
  font-weight: 600 !important;
  letter-spacing: 1px !important;
  font-size: 12px !important;
  text-transform: uppercase !important;
  border: none !important;
  background: transparent !important;
}
.tab-nav button.selected {
  color: var(--purple) !important;
  border-bottom: 2px solid var(--purple) !important;
}

/* Dropdown */
.gr-dropdown, .gr-textbox input {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-dim) !important;
  color: var(--text-bright) !important;
  border-radius: 8px !important;
}

/* Plot container — let matplotlib's dark bg show through */
.plot-container, .gr-plot {
  background: transparent !important;
  border: none !important;
}
"""

# ── Service bootstrap ────────────────────────────────────────────────────────

def _build_services():
    detector = FaceDetector()
    embedder = ArcFaceEmbedder(detector)
    squeezers = [
        BitDepthSqueezer(bits=4),
        MedianFilterSqueezer(kernel=3),
        NonLocalMeansSqueezer(strength=11),
    ]
    squeeze_detector = SqueezeDetector(embedder=embedder, squeezers=squeezers)
    db = IdentityDatabase(embedder=embedder, dataset_path="dataset")
    db.load()
    return detector, embedder, squeeze_detector, db


DETECTOR, EMBEDDER, SQUEEZE_DETECTOR, DB = _build_services()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _detect_face(img: np.ndarray) -> Face | None:
    """Run the (already-bootstrapped) face detector and wrap the first hit."""
    if img is None:
        return None
    raw = DETECTOR.detect(img)
    if not raw:
        return None
    f = raw[0]
    return Face(bbox=f.bbox, kps=f.kps)


def _img_array(pil_or_array) -> np.ndarray | None:
    if pil_or_array is None:
        return None
    if isinstance(pil_or_array, np.ndarray):
        return pil_or_array
    return np.array(pil_or_array)


# ── Matplotlib figures ───────────────────────────────────────────────────────

_BG = "#0a0a14"
_PANEL = "#14142a"

_STAGE_COLORS = {
    "original": "#94a3b8",
    "attacked": "#ef4444",
    "bit":      "#fbbf24",
    "median":   "#10b981",
    "nlm":      "#38bdf8",
}
_STAGE_LABELS = {
    "original": "ORIGINAL",
    "attacked": "ADV. GLASSES",
    "bit":      "BIT SQUEEZED",
    "median":   "MEDIAN FILTER",
    "nlm":      "NON-LOCAL MEANS",
}
_PIPELINE_KEYS = ["original", "attacked", "bit", "median", "nlm"]


def _style_image_axis(ax, color: str, title: str, sim: float | None) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("left", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["top"].set_edgecolor(color)
    ax.spines["top"].set_linewidth(3)
    ax.set_title(title, color=color, fontsize=10, fontweight="700",
                 pad=8, loc="left")
    if sim is not None:
        ax.set_xlabel(f"sim  {sim:+.2f}", color=color, fontsize=15,
                      fontweight="800", labelpad=8)


def _render_pipeline_figure(
    imgs: dict[str, np.ndarray | None],
    sims: dict[str, float],
) -> plt.Figure:
    fig, axes = plt.subplots(1, len(_PIPELINE_KEYS), figsize=(16.5, 3.6))
    fig.patch.set_facecolor(_BG)

    for ax, key in zip(axes, _PIPELINE_KEYS):
        color = _STAGE_COLORS[key]
        ax.set_facecolor(_PANEL)
        img = imgs.get(key)
        if img is not None:
            ax.imshow(img)
        sim = sims.get(key)
        # show sim under panel only if we have a real value (non-zero) and image present
        show_sim = sim if (img is not None and sim is not None and sim != 0.0) else None
        _style_image_axis(ax, color, _STAGE_LABELS[key], show_sim)

    fig.tight_layout(pad=1.4)
    return fig


def _render_similarity_bars(sims: dict[str, float]) -> plt.Figure:
    keys   = _PIPELINE_KEYS
    labels = [_STAGE_LABELS[k] for k in keys]
    values = [sims.get(k, 0.0) for k in keys]
    colors = [_STAGE_COLORS[k] for k in keys]

    fig, ax = plt.subplots(figsize=(11, 3.2))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    y_pos = list(range(len(keys)))
    bars = ax.barh(y_pos, values, color=colors, height=0.55, alpha=0.92,
                   edgecolor="none")
    ax.axvline(x=0.50, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlim(-0.05, 1.10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="#94a3b8", fontsize=10,
                       fontweight="600")
    ax.tick_params(axis="x", colors="#475569", labelsize=9)
    ax.text(0.51, -0.95, "threshold 0.50", color="#ef4444",
            fontsize=8, alpha=0.75, fontweight="600")
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, val, color in zip(bars, values, colors):
        ax.text(bar.get_width() + 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}", va="center", color=color,
                fontsize=13, fontweight="800")
    fig.tight_layout()
    return fig


def _render_verification_figure(rows: list[dict]) -> plt.Figure:
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 3.0),
                             gridspec_kw={"width_ratios": [1, 1.6, 1]})
    fig.patch.set_facecolor(_BG)
    if n == 1:
        axes = [axes]

    for row, axrow in zip(rows, axes):
        ax_a, ax_bar, ax_b = axrow
        color = row["color"]

        for ax, img, ttl in [(ax_a, row["img_a"], "REF A"),
                             (ax_b, row["img_b"], "TEST B")]:
            ax.set_facecolor(_PANEL)
            if img is not None:
                ax.imshow(img)
            ax.set_title(ttl, color="#94a3b8", fontsize=9, fontweight="700")
            ax.set_xticks([]); ax.set_yticks([])
            for side in ("left", "right", "bottom"):
                ax.spines[side].set_visible(False)
            ax.spines["top"].set_edgecolor(color)
            ax.spines["top"].set_linewidth(2.5)

        ax_bar.set_facecolor(_BG)
        ax_bar.barh([0], [row["sim"]], color=color, height=0.45,
                    alpha=0.9, edgecolor="none")
        ax_bar.axvline(0.50, color="#ef4444", linestyle="--",
                       linewidth=1.2, alpha=0.7)
        ax_bar.set_xlim(-0.05, 1.10); ax_bar.set_yticks([])
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        ax_bar.tick_params(colors="#475569", labelsize=9)
        ax_bar.set_title(row["label"], color=color, fontsize=11,
                         fontweight="800", pad=10)
        ax_bar.text(row["sim"] + 0.015, 0,
                    f'{row["sim"]:+.2f}   {row["verdict"]}',
                    va="center", color=color, fontsize=12,
                    fontweight="800")

    fig.tight_layout(pad=1.6)
    return fig


# ── Verdict banners ──────────────────────────────────────────────────────────

def _verdict_html(is_adversarial: bool, max_shift: float) -> str:
    if is_adversarial:
        return f"""
        <div style="background:linear-gradient(135deg,rgba(239,68,68,0.18),rgba(239,68,68,0.04));
                    border:1px solid #ef4444;border-radius:12px;padding:18px 26px;
                    display:flex;align-items:center;gap:14px;
                    box-shadow:0 0 32px rgba(239,68,68,0.32),
                               inset 0 1px 0 rgba(255,255,255,0.04);">
          <span style="font-size:30px;">&#128680;</span>
          <div>
            <div style="font-size:15px;color:#fca5a5;font-weight:800;
                        letter-spacing:1.8px;text-transform:uppercase;">
              Adversarial Input Detected
            </div>
            <div style="font-size:11px;color:#94a3b8;margin-top:4px;letter-spacing:0.4px;">
              Embedding shift {max_shift:.3f} &mdash; above threshold (0.50).
              Attack neutralised by Feature Squeezing.
            </div>
          </div>
        </div>"""
    return f"""
    <div style="background:linear-gradient(135deg,rgba(16,185,129,0.18),rgba(16,185,129,0.04));
                border:1px solid #10b981;border-radius:12px;padding:18px 26px;
                display:flex;align-items:center;gap:14px;
                box-shadow:0 0 32px rgba(16,185,129,0.30),
                           inset 0 1px 0 rgba(255,255,255,0.04);">
      <span style="font-size:30px;">&#9989;</span>
      <div>
        <div style="font-size:15px;color:#6ee7b7;font-weight:800;
                    letter-spacing:1.8px;text-transform:uppercase;">
          Clean Input
        </div>
        <div style="font-size:11px;color:#94a3b8;margin-top:4px;letter-spacing:0.4px;">
          Embedding shift {max_shift:.3f} &mdash; below threshold (0.50).
        </div>
      </div>
    </div>"""


# ── Tab 1 handlers ───────────────────────────────────────────────────────────

_EMPTY_KEYS = _PIPELINE_KEYS


def on_identity_select(name: str):
    identity = DB.get(name)
    empty_sims = {k: 0.0 for k in _EMPTY_KEYS}
    empty_imgs = {k: None for k in _EMPTY_KEYS}
    empty_imgs["original"] = identity.photo
    return (
        identity.photo,
        identity.embedding,
        identity.photo,
        None,
        _render_pipeline_figure(empty_imgs, empty_sims),
        _render_similarity_bars(empty_sims),
        gr.update(visible=False),
    )


def on_attack(original_img, target_embed, bits, kernel):
    if original_img is None or target_embed is None:
        return gr.update(), gr.update(), None, gr.update(visible=False)

    face = _detect_face(original_img)
    if face is None:
        return gr.update(), gr.update(), None, gr.update(
            value='<div style="color:#fca5a5;font-size:12px;">'
                  'No face detected in image.</div>',
            visible=True,
        )

    attacker = GlassesAttacker(epsilon=0.15)
    attacked = attacker.apply(original_img, face)

    empty_sims = {k: 0.0 for k in _EMPTY_KEYS}
    imgs = {k: None for k in _EMPTY_KEYS}
    imgs["original"] = original_img
    imgs["attacked"] = attacked
    return (
        _render_pipeline_figure(imgs, empty_sims),
        _render_similarity_bars(empty_sims),
        attacked,
        gr.update(visible=False),
    )


def on_squeeze_detect(original_img, attacked_img, target_embed, bits, kernel):
    if attacked_img is None or target_embed is None or original_img is None:
        return gr.update(), gr.update(), gr.update(visible=False)

    squeezers = [
        BitDepthSqueezer(bits=int(bits)),
        MedianFilterSqueezer(kernel=int(kernel)),
        NonLocalMeansSqueezer(strength=11),
    ]
    local_detector = SqueezeDetector(EMBEDDER, squeezers)
    result = local_detector.detect(target_embed, original_img, attacked_img)

    imgs = {
        "original": original_img,
        "attacked": attacked_img,
        "bit":      result.squeezed_imgs["bit"],
        "median":   result.squeezed_imgs["median"],
        "nlm":      result.squeezed_imgs["nlm"],
    }
    return (
        _render_pipeline_figure(imgs, result.sims),
        _render_similarity_bars(result.sims),
        gr.update(value=_verdict_html(result.is_adversarial, result.max_shift),
                  visible=True),
    )


def on_register(name: str, photo1, photo2, db_state):
    img1 = _img_array(photo1)
    img2 = _img_array(photo2)
    if img1 is None or img2 is None or not name or not name.strip():
        return db_state, gr.update()
    try:
        DB.register(name.strip(), img1, img2)
    except ValueError:
        return db_state, gr.update()
    return db_state, gr.update(choices=DB.names(), value=name.strip())


# ── Tab 2 handler ────────────────────────────────────────────────────────────

def on_verify(photo_a, photo_b):
    img_a = _img_array(photo_a)
    img_b = _img_array(photo_b)
    if img_a is None or img_b is None:
        return gr.update(), (
            '<div style="color:#fca5a5;font-size:12px;">'
            'Upload both photos first.</div>'
        )

    embed_a = EMBEDDER.embed(img_a)
    embed_b_clean = EMBEDDER.embed(img_b)
    if embed_a is None or embed_b_clean is None:
        return gr.update(), (
            '<div style="color:#fca5a5;font-size:12px;">'
            'Could not detect a face in one of the images.</div>'
        )

    face_b = _detect_face(img_b)
    if face_b is None:
        return gr.update(), (
            '<div style="color:#fca5a5;font-size:12px;">'
            'No face found in Photo B for attack.</div>'
        )

    attacked_b = GlassesAttacker(epsilon=0.15).apply(img_b, face_b)
    embed_b_attacked = EMBEDDER.embed(attacked_b)

    squeezed_b = MedianFilterSqueezer(kernel=3).squeeze(attacked_b)
    embed_b_squeezed = EMBEDDER.embed(squeezed_b)

    def _sim(a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))

    sim_clean    = _sim(embed_a, embed_b_clean)
    sim_attacked = _sim(embed_a, embed_b_attacked)
    sim_squeezed = _sim(embed_a, embed_b_squeezed)

    rows = [
        {
            "label":   "CLEAN A  vs  CLEAN B",
            "img_a":   img_a, "img_b": img_b,
            "sim":     sim_clean,
            "verdict": "SAME PERSON" if sim_clean > 0.50 else "DIFFERENT",
            "color":   "#10b981" if sim_clean > 0.50 else "#ef4444",
        },
        {
            "label":   "CLEAN A  vs  ATTACKED B",
            "img_a":   img_a, "img_b": attacked_b,
            "sim":     sim_attacked,
            "verdict": "SAME PERSON" if sim_attacked > 0.50 else "IDENTITY LOST",
            "color":   "#10b981" if sim_attacked > 0.50 else "#ef4444",
        },
        {
            "label":   "CLEAN A  vs  SQUEEZED-ATTACKED B",
            "img_a":   img_a, "img_b": squeezed_b,
            "sim":     sim_squeezed,
            "verdict": "IDENTITY RECOVERED" if sim_squeezed > 0.50 else "NOT RECOVERED",
            "color":   "#10b981" if sim_squeezed > 0.50 else "#ef4444",
        },
    ]
    return _render_verification_figure(rows), ""


# ── App layout ───────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="Feature Squeezing Demo",
                   theme=gr.themes.Base()) as demo:

        # Header
        gr.HTML("""
        <div style="text-align:center;padding:24px 0 12px;">
          <div style="font-size:30px;font-weight:900;color:#a78bfa;
                      letter-spacing:6px;text-transform:uppercase;
                      text-shadow:0 0 24px rgba(167,139,250,0.45);">
            &#128737; Feature Squeezing
          </div>
          <div style="font-size:11px;color:#64748b;margin-top:6px;
                      letter-spacing:3px;text-transform:uppercase;">
            Detecting Adversarial Examples in Deep Neural Networks
          </div>
          <div style="display:flex;gap:10px;justify-content:center;margin-top:14px;flex-wrap:wrap;">
            <span style="background:rgba(239,68,68,0.10);
                         border:1px solid rgba(239,68,68,0.35);
                         border-radius:20px;padding:5px 14px;
                         font-size:10px;color:#fca5a5;letter-spacing:1.2px;
                         text-transform:uppercase;font-weight:600;">
              &#9876; Sharif et al. &middot; CCS 2016
            </span>
            <span style="background:rgba(16,185,129,0.10);
                         border:1px solid rgba(16,185,129,0.35);
                         border-radius:20px;padding:5px 14px;
                         font-size:10px;color:#6ee7b7;letter-spacing:1.2px;
                         text-transform:uppercase;font-weight:600;">
              &#128737; Xu et al. &middot; NDSS 2018
            </span>
          </div>
        </div>
        """)

        state_target_embed = gr.State(value=None)
        state_original_img = gr.State(value=None)
        state_attacked_img = gr.State(value=None)
        state_db           = gr.State(value=None)

        with gr.Tabs():
            # ── TAB 1 ────────────────────────────────────────────────────────
            with gr.Tab("\u2694 Pipeline Demo"):
                with gr.Row():
                    # Sidebar
                    with gr.Column(scale=1, min_width=200):
                        identity_dd = gr.Dropdown(
                            choices=DB.names(),
                            label="Target Identity",
                            value=DB.names()[0] if DB.names() else None,
                        )
                        target_photo = gr.Image(
                            label="Selected", height=140, interactive=False,
                            show_download_button=False,
                        )
                        gr.HTML(
                            '<hr style="border:none;border-top:1px solid '
                            'rgba(255,255,255,0.06);margin:12px 0;">'
                        )
                        gr.HTML(
                            '<div style="font-size:9px;color:#a78bfa;'
                            'text-transform:uppercase;letter-spacing:2px;'
                            'margin-bottom:8px;font-weight:700;">'
                            '&#43; Register Face</div>'
                        )
                        reg_photo1 = gr.Image(label="Photo 1", height=80, type="numpy")
                        reg_photo2 = gr.Image(label="Photo 2", height=80, type="numpy")
                        reg_name   = gr.Textbox(label="Name", placeholder="Your name")
                        reg_btn    = gr.Button("\u002B Register",
                                               elem_classes=["btn-register"])
                        gr.HTML(
                            '<hr style="border:none;border-top:1px solid '
                            'rgba(255,255,255,0.06);margin:12px 0;">'
                        )
                        attack_btn = gr.Button("\u2694 Launch Attack",
                                               elem_classes=["btn-attack"])
                        defend_btn = gr.Button("\U0001F6E1 Squeeze + Detect",
                                               elem_classes=["btn-defend"])

                    # Main panel
                    with gr.Column(scale=4):
                        with gr.Group(elem_classes=["squeezer-card"]):
                            gr.HTML(
                                '<div style="font-size:10px;color:#a78bfa;'
                                'text-transform:uppercase;letter-spacing:3px;'
                                'text-align:center;margin-bottom:14px;'
                                'font-weight:700;">'
                                '&#9881; Squeezer Controls</div>'
                            )
                            with gr.Row():
                                bits_slider = gr.Slider(
                                    1, 8, value=4, step=1,
                                    label="Bit Depth  (1 = 2 levels … 8 = 256 levels)",
                                )
                                kernel_slider = gr.Slider(
                                    3, 7, value=3, step=2,
                                    label="Median Kernel  (3 / 5 / 7)",
                                )

                        pipeline_plot = gr.Plot(show_label=False)
                        sim_plot      = gr.Plot(show_label=False)
                        verdict_html  = gr.HTML(visible=False)

                # Wire Tab 1 events
                identity_dd.change(
                    fn=on_identity_select,
                    inputs=[identity_dd],
                    outputs=[target_photo, state_target_embed, state_original_img,
                             state_attacked_img, pipeline_plot, sim_plot,
                             verdict_html],
                )
                attack_btn.click(
                    fn=on_attack,
                    inputs=[state_original_img, state_target_embed,
                            bits_slider, kernel_slider],
                    outputs=[pipeline_plot, sim_plot, state_attacked_img,
                             verdict_html],
                )
                defend_btn.click(
                    fn=on_squeeze_detect,
                    inputs=[state_original_img, state_attacked_img,
                            state_target_embed, bits_slider, kernel_slider],
                    outputs=[pipeline_plot, sim_plot, verdict_html],
                )
                reg_btn.click(
                    fn=on_register,
                    inputs=[reg_name, reg_photo1, reg_photo2, state_db],
                    outputs=[state_db, identity_dd],
                )

                # Initialise the visible state on first load
                demo.load(
                    fn=on_identity_select,
                    inputs=[identity_dd],
                    outputs=[target_photo, state_target_embed, state_original_img,
                             state_attacked_img, pipeline_plot, sim_plot,
                             verdict_html],
                )

            # ── TAB 2 ────────────────────────────────────────────────────────
            with gr.Tab("\U0001F50D Verify Identity"):
                gr.HTML("""
                <div style="color:#94a3b8;font-size:13px;padding:14px 4px 18px;
                            line-height:1.55;">
                  Upload two photos of the
                  <strong style="color:#a78bfa;">same person</strong>.
                  See how the adversarial glasses break recognition &mdash;
                  and how squeezing restores it.
                </div>
                """)
                with gr.Row():
                    verify_photo_a = gr.Image(
                        label="Photo A — Reference",
                        type="numpy", height=220,
                    )
                    verify_photo_b = gr.Image(
                        label="Photo B — Test (same person)",
                        type="numpy", height=220,
                    )
                verify_btn = gr.Button(
                    "\U0001F50D Run Verification",
                    elem_classes=["btn-defend"],
                )
                verify_error = gr.HTML()
                verify_plot  = gr.Plot(show_label=False)

                verify_btn.click(
                    fn=on_verify,
                    inputs=[verify_photo_a, verify_photo_b],
                    outputs=[verify_plot, verify_error],
                )

        # Footer
        gr.HTML("""
        <div style="text-align:center;color:#475569;font-size:10px;
                    letter-spacing:1.5px;text-transform:uppercase;
                    margin-top:24px;padding:16px 0;">
          HUP Seminar &middot;  &amp;  &middot;
          ArcFace + Feature Squeezing
        </div>
        """)

    return demo


if __name__ == "__main__":
    # LOCAL-ONLY: share=True bypasses the localhost-not-accessible error on this machine.
    # REVERT BEFORE HF DEPLOY: change back to `build_app().launch()` — HF Spaces handles networking.
    build_app().launch(share=True, show_api=False)

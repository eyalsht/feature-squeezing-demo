"""Generate the animated README assets for the Feature Squeezing demo.

Builds two looping GIFs from the real demo faces in ``dataset/``:

* ``pipeline.gif`` — the headline story: Original → Adv. Glasses → Bit-Depth →
  Median, with cosine-similarity bars filling in and the verdict banner flipping
  to ADVERSARIAL DETECTED.
* ``squeeze.gif`` — a bit-depth sweep (8→1 bits) showing how squeezing collapses
  the adversarial high-frequency perturbation.

The similarity numbers are illustrative values consistent with the design spec
(ArcFace is not run here — this script only needs Pillow + numpy so the assets
can be regenerated in CI without the 300 MB model download).

    python docs/assets/build_gifs.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset"
OUT = ROOT / "docs" / "assets"

# ── palette (mirrors app.py CSS) ─────────────────────────────────────────────
BG = (10, 10, 20)
SURFACE = (20, 20, 42)
CARD = (24, 24, 48)
DIM = (100, 116, 139)
BODY = (148, 163, 184)
BRIGHT = (226, 232, 240)
PURPLE = (167, 139, 250)
RED = (239, 68, 68)
GREEN = (16, 185, 129)
AMBER = (251, 191, 36)

FONTS = ROOT.parent / "skills"  # unused fallback
FDIR_CANVAS = Path("/mnt/skills/examples/canvas-design/canvas-fonts")
FDIR_SYS = Path("/usr/share/fonts/truetype/liberation")


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = [FDIR_CANVAS / name, FDIR_SYS / name]
    for c in candidates:
        if c.exists():
            return ImageFont.truetype(str(c), size)
    return ImageFont.load_default()


F_KICKER = font("InstrumentSans-Bold.ttf", 15)
F_TITLE = font("InstrumentSans-Bold.ttf", 30)
F_LABEL = font("WorkSans-Bold.ttf", 16)
F_SUB = font("WorkSans-Regular.ttf", 13)
F_NUM = font("LiberationMono-Bold.ttf", 22)
F_BANNER = font("InstrumentSans-Bold.ttf", 24)


# ── image ops ────────────────────────────────────────────────────────────────
def load_face(name: str, box: tuple[int, int, int], size: int = 190) -> np.ndarray:
    x, y, s = box
    im = Image.open(DATA / name).convert("RGB").crop((x, y, x + s, y + s))
    return np.asarray(im.resize((size, size), Image.LANCZOS)).astype(np.float32)


def glasses_noise(face: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    h, w, _ = face.shape
    mask = np.zeros((h, w, 1), np.float32)
    y1, y2 = int(0.33 * h), int(0.51 * h)
    x1, x2 = int(0.14 * w), int(0.86 * w)
    mask[y1:y2, x1:x2] = 1.0
    noise = rng.uniform(-eps * 255, eps * 255, face.shape).astype(np.float32)
    return np.clip(face + noise * mask, 0, 255)


def bit_depth(face: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** bits
    step = 256 // levels
    return np.clip((face.astype(np.uint16) // step) * step, 0, 255).astype(np.float32)


def median(face: np.ndarray, k: int = 5) -> np.ndarray:
    im = Image.fromarray(face.astype(np.uint8)).filter(ImageFilter.MedianFilter(k))
    return np.asarray(im).astype(np.float32)


def rounded(arr: np.ndarray, radius: int = 16) -> Image.Image:
    im = Image.fromarray(arr.astype(np.uint8))
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, im.size[0] - 1, im.size[1] - 1],
                                           radius=radius, fill=255)
    im.putalpha(mask)
    return im


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def bg_canvas(w: int, h: int) -> Image.Image:
    base = Image.new("RGB", (w, h), BG)
    # soft purple glow top-left, red glow bottom-right
    glow = Image.new("RGB", (w, h), BG)
    gd = ImageDraw.Draw(glow, "RGBA")
    gd.ellipse([-w * 0.3, -h * 0.5, w * 0.6, h * 0.7], fill=(167, 139, 250, 26))
    gd.ellipse([w * 0.5, h * 0.4, w * 1.3, h * 1.5], fill=(239, 68, 68, 18))
    glow = glow.filter(ImageFilter.GaussianBlur(80))
    return Image.blend(base, glow, 0.55)


# ── panel drawing ────────────────────────────────────────────────────────────
def draw_panel(canvas: Image.Image, d: ImageDraw.ImageDraw, x: int, y: int,
               face: np.ndarray, accent, label: str, sub: str,
               sim: float, fill: float, active: bool, size: int = 190):
    # accent top bar + border
    a = accent if active else tuple(int(c * 0.4) for c in accent)
    d.rounded_rectangle([x - 6, y - 6, x + size + 6, y + size + 92],
                        radius=20, fill=CARD,
                        outline=a, width=2 if active else 1)
    d.rounded_rectangle([x + 6, y - 6, x + size - 6, y + 2], radius=4, fill=a)
    canvas.paste(rounded(face), (x, y), rounded(face))

    d.text((x, y + size + 12), label, font=F_LABEL,
           fill=BRIGHT if active else DIM)
    d.text((x, y + size + 33), sub, font=F_SUB, fill=DIM)

    # sim bar
    by = y + size + 58
    bw = size
    d.rounded_rectangle([x, by, x + bw, by + 10], radius=5, fill=SURFACE)
    if fill > 0:
        d.rounded_rectangle([x, by, x + int(bw * sim * fill), by + 10],
                            radius=5, fill=a)
    shown = sim * fill
    d.text((x + bw, by - 26), f"{shown:0.2f}", font=F_NUM, fill=a,
           anchor="ra")


def arrow(d: ImageDraw.ImageDraw, x: int, y: int, lit: bool):
    c = PURPLE if lit else (60, 60, 90)
    d.line([x, y, x + 22, y], fill=c, width=3)
    d.polygon([(x + 22, y - 6), (x + 34, y), (x + 22, y + 6)], fill=c)


# ── pipeline.gif ─────────────────────────────────────────────────────────────
def build_pipeline():
    W, H = 960, 470
    rng = np.random.default_rng(7)
    orig = load_face("ron_1.jpg", (300, 100, 170))

    sims = [0.81, 0.19, 0.74, 0.78]
    accents = [PURPLE, RED, AMBER, GREEN]
    labels = ["ORIGINAL", "ADV. GLASSES", "BIT-DEPTH · 3", "MEDIAN · 5×5"]
    subs = ["clean reference", "Sharif et al. 2016",
            "8 → 3 bits", "scipy median"]

    px0, py = 48, 96
    gap = 224
    size = 190
    frames = []
    N = 56
    for f in range(N):
        cv = bg_canvas(W, H)
        d = ImageDraw.Draw(cv, "RGBA")

        # header
        d.text((48, 30), "FEATURE  SQUEEZING", font=F_KICKER, fill=PURPLE)
        d.text((48, 48), "Detecting adversarial faces by squeezing pixel space",
               font=F_SUB, fill=BODY)

        # per-panel reveal + fill timing
        reveal = [2, 12, 24, 28]
        faces = [
            orig,
            glasses_noise(orig, 0.16, rng),
            bit_depth(glasses_noise(orig, 0.16, np.random.default_rng(7)), 3),
            median(glasses_noise(orig, 0.16, np.random.default_rng(7)), 5),
        ]
        lit_to = 0
        for i in range(4):
            fill = ease((f - reveal[i]) / 9.0)
            active = f >= reveal[i]
            x = px0 + i * gap
            draw_panel(cv, d, x, py, faces[i], accents[i], labels[i],
                       subs[i], sims[i], fill, active, size)
            if active:
                lit_to = i
            if i < 3:
                arrow(d, x + size + 8, py + size // 2, f >= reveal[i + 1])

        # verdict banner
        bx0, bx1 = 48, W - 48
        byb = 416
        detected = f >= 38
        if detected:
            pulse = 0.5 + 0.5 * np.sin((f - 38) * 0.55)
            col = RED
            d.rounded_rectangle([bx0, byb, bx1, byb + 44], radius=14,
                                fill=(60, 18, 22), outline=col,
                                width=2 + int(2 * pulse))
            txt = "ADVERSARIAL DETECTED"
            # drawn warning triangle (no colour-emoji font available)
            tx, ty = bx0 + 30, byb + 12
            d.polygon([(tx, ty + 20), (tx + 12, ty), (tx + 24, ty + 20)],
                      fill=col)
            d.line([(tx + 12, ty + 6), (tx + 12, ty + 13)], fill=(60, 18, 22),
                   width=2)
            d.ellipse([tx + 11, ty + 15, tx + 13, ty + 17], fill=(60, 18, 22))
            d.text((bx0 + 68, byb + 10), txt, font=F_BANNER, fill=(255, 220, 220))
            d.text((bx1 - 22, byb + 15),
                   "embedding shift  0.59  >  0.50", font=F_LABEL,
                   fill=(255, 180, 180), anchor="ra")
        else:
            col = DIM
            d.rounded_rectangle([bx0, byb, bx1, byb + 44], radius=14,
                                fill=SURFACE, outline=(70, 70, 100), width=1)
            dots = "." * (1 + (f // 3) % 3)
            d.text((bx0 + 22, byb + 11), f"ANALYZING EMBEDDINGS{dots}",
                   font=F_LABEL, fill=BODY)

        frames.append(cv.convert("P", palette=Image.ADAPTIVE, colors=128))

    frames += [frames[-1]] * 10  # hold
    frames[0].save(OUT / "pipeline.gif", save_all=True,
                   append_images=frames[1:], duration=90, loop=0, disposal=2,
                   optimize=True)
    print("wrote", OUT / "pipeline.gif")


# ── squeeze.gif ──────────────────────────────────────────────────────────────
def build_squeeze():
    W, H = 620, 430
    size = 230
    orig = load_face("ron_1.jpg", (300, 100, 170), size)
    attacked = glasses_noise(orig, 0.18, np.random.default_rng(3))

    seq = [8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8]
    frames = []
    for bits in seq:
        cv = bg_canvas(W, H)
        d = ImageDraw.Draw(cv, "RGBA")
        d.text((40, 28), "BIT-DEPTH  SQUEEZING", font=F_KICKER, fill=AMBER)
        d.text((40, 47), "Fewer levels collapse the adversarial perturbation",
               font=F_SUB, fill=BODY)

        sq = bit_depth(attacked, bits)
        lx = (W - 2 * size - 40) // 2
        # attacked (static) left, squeezed right
        d.rounded_rectangle([lx - 6, 84, lx + size + 6, 84 + size + 46],
                            radius=18, fill=CARD, outline=RED, width=2)
        cv.paste(rounded(attacked), (lx, 90), rounded(attacked))
        d.text((lx, 90 + size + 8), "ADVERSARIAL  ·  8-bit", font=F_SUB, fill=RED)

        rx = lx + size + 40
        d.rounded_rectangle([rx - 6, 84, rx + size + 6, 84 + size + 46],
                            radius=18, fill=CARD, outline=AMBER, width=2)
        cv.paste(rounded(sq), (rx, 90), rounded(sq))
        d.text((rx, 90 + size + 8),
               f"SQUEEZED  ·  {bits}-bit ({2 ** bits} levels)",
               font=F_SUB, fill=AMBER)

        # bit gauge
        gy = 90 + size + 32
        d.text((W // 2, gy), f"{bits}  bits", font=F_NUM, fill=BRIGHT, anchor="ma")
        frames.append(cv.convert("P", palette=Image.ADAPTIVE, colors=128))

    frames[0].save(OUT / "squeeze.gif", save_all=True,
                   append_images=frames[1:], duration=260, loop=0, disposal=2,
                   optimize=True)
    print("wrote", OUT / "squeeze.gif")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_pipeline()
    build_squeeze()

# Feature Squeezing Live Demo — Design Spec
**Date:** 2026-05-13  
**Project:** HUP Seminar — Adversarial ML Demo  
**Platform:** Hugging Face Spaces (Gradio)

---

## 1. Purpose

A live, interactive web demo deployed on Hugging Face Spaces that visualises the attack-and-defense story between two papers:

- **Attacker:** *Accessorize to a Crime* (Sharif et al., CCS 2016) — physical adversarial glasses that fool face recognition
- **Defender:** *Feature Squeezing* (Xu et al., NDSS 2018) — detects adversarial inputs by squeezing pixel space and measuring embedding shift

The audience is a seminar class. The goal is a single URL they open, see the attack break face recognition, then watch squeezing recover and detect it — no setup, no code visible.

---

## 2. Architecture

### Hosting
Hugging Face Spaces (free tier, CPU-only, public URL). Deployed from a GitHub repo via HF Spaces git integration.

### Repository structure
```
adversarial-demo/
├── app.py              # Gradio UI — all tabs, layout, event wiring
├── squeezers.py        # bit_depth_reduce(), median_filter_squeeze()
├── attack.py           # get_glasses_mask(), apply_attack()
├── detector.py         # load_arcface(), get_embedding(), cosine_sim(), detect()
├── dataset.py          # load_dataset(), register_identity(), session state helpers
├── dataset/            # 6–8 pre-loaded identities (Creative Commons face images)
│   ├── person_a.jpg
│   └── ...
├── requirements.txt
└── README.md           # HF Spaces config header (title, emoji, theme colour)
```

### Tech stack
| Layer | Tool | Reason |
|---|---|---|
| UI framework | Gradio 4.x (Blocks API) | Native HF Spaces support, no extra server |
| Face model | `insightface` (ArcFace, buffalo_l) | Pretrained, CPU-compatible, one-line embed |
| Face detection | `insightface` built-in detector | Needed for glasses landmark placement |
| Image ops | `numpy` + `Pillow` + `scipy.ndimage` | Bit reduction + median filter |
| Hosting | Hugging Face Spaces (free) | Public URL, zero ops |

### Constraints
- CPU only — no GPU on free tier; `insightface` runs fine on CPU for demo latency
- No persistent storage — registered faces live in `gr.State()` per session
- Model weights auto-downloaded by `insightface` on first Space boot (~300 MB, cached)

---

## 3. Visual Design

**Theme:** Bold & Modern — deep navy/indigo background (`#0f0f1a` → `#1a1a2e`), purple accents (`#a78bfa`), red for attack (`#ef4444`), green for safe (`#10b981`), amber for bit squeeze (`#fbbf24`).

**Custom CSS** applied via `gr.Blocks(css=...)` to override Gradio defaults.

---

## 4. UI Layout

### Tab 1 — Pipeline Demo

Two-column layout:

**Left sidebar (160px, slim):**
- Identity dropdown (pre-loaded dataset)
- Avatar strip (clickable identity thumbnails)
- "Add Your Face" upload widget + name field + Register button
- `⚔ Launch Attack` button (red)
- `🛡 Squeeze + Detect` button (green)

**Main panel:**
1. **Squeezer Controls** — hero element, glowing purple card spanning full width:
   - Bit Depth slider (1–8 bits) with large numeric readout + one-line description
   - Median Kernel slider (3×3 / 5×5 / 7×7, odd only) with large numeric readout + one-line description
2. **Pipeline Strip** — 4 image cards in a row with labeled arrows between them:
   - Original → Adv. Glasses → Bit Squeezed → Median Filtered
   - Cosine similarity score displayed under each card
3. **Verdict banner** — full-width, 🚨 ADVERSARIAL DETECTED (red) or ✅ CLEAN INPUT (green)

### Tab 2 — Verify Identity

**Purpose:** Show impersonation attack — same person recognised clean, not recognised under attack, recovered after squeezing.

**Layout:**
- Two upload slots: Photo A (reference) + Photo B (test, same person)
- `Run Verification` button
- Three result rows:
  1. Clean A vs Clean B → sim score → ✅ SAME PERSON
  2. Clean A vs Attacked B → sim score → ❌ IDENTITY LOST
  3. Clean A vs Squeezed-Attacked B → sim score → ✅ IDENTITY RECOVERED
- Each row shows a small face thumbnail pair + similarity bar + verdict chip

---

## 5. Feature List

| # | Feature | Tab | Notes |
|---|---|---|---|
| 1 | Identity selector dropdown | 1 | 3 pre-loaded CC-licensed identities |
| 2 | Avatar strip | 1 | Visual quick-select; active identity highlighted |
| 3 | Register face | 1 | Upload 2 photos + name → average ArcFace embed → added to `gr.State` (1 live identity max) |
| 4 | Launch Attack | 1 | Landmark detect → glasses mask → uniform noise overlay |
| 5 | Bit Depth slider | 1 | 1–8 bits; reduces pixel values to 2^n levels |
| 6 | Median Kernel slider | 1 | 3×3 / 5×5 / 7×7 (odd only); scipy median filter |
| 7 | Pipeline strip | 1 | 4 panels + cosine sim scores; updates on Squeeze+Detect |
| 8 | Verdict banner | 1 | max(shift_bit, shift_median) > 0.50 → DETECTED |
| 9 | Same-person verification | 2 | 3-row comparison: clean / attacked / squeezed |

**Out of scope:**
- Persistent storage across sessions
- Real FGSM/PGD optimised attack (simulated noise only)
- Multi-identity comparison in a single run

---

## 6. Data Flow

### Tab 1 — Pipeline
```
select/upload identity
  → ArcFace embed(face) → stored in gr.State as embed_target

[Launch Attack]
  → detect landmarks (insightface)
  → glasses_mask = numpy rectangle over eye region
  → noise = np.random.uniform(-ε, ε, mask.shape)
  → attacked_img = original + noise * mask
  → display attacked_img in pipeline strip slot 2

[Squeeze + Detect]
  → bit_img   = bit_depth_reduce(attacked_img, bits=slider_bits)
  → med_img   = median_filter_squeeze(attacked_img, kernel=slider_kernel)
  → embed_attacked  = ArcFace embed(attacked_img)
  → embed_bit       = ArcFace embed(bit_img)
  → embed_med       = ArcFace embed(med_img)
  → sim_original = cosine(embed_target, embed_original)
  → sim_attacked = cosine(embed_target, embed_attacked)
  → sim_bit      = cosine(embed_target, embed_bit)
  → sim_med      = cosine(embed_target, embed_med)
  → shift = max(|sim_original - sim_bit|, |sim_original - sim_med|)
  → verdict = DETECTED if shift > 0.50 else CLEAN
  → update: 4 pipeline images + 4 sim scores + verdict banner
```

### Tab 2 — Verify Identity
```
upload Photo A + Photo B
  → embed_A = ArcFace embed(A)
  → embed_B_clean    = ArcFace embed(B)
  → embed_B_attacked = ArcFace embed(apply_attack(B))
  → embed_B_squeezed = ArcFace embed(median_filter_squeeze(apply_attack(B), kernel=3))
  → sim_clean    = cosine(embed_A, embed_B_clean)     → ✅ if > 0.50
  → sim_attacked = cosine(embed_A, embed_B_attacked)  → ❌ if < 0.50
  → sim_squeezed = cosine(embed_A, embed_B_squeezed)  → ✅ if > 0.50
  → display 3 result rows
```

---

## 7. Module Contracts

### `squeezers.py`
```python
def bit_depth_reduce(img: np.ndarray, bits: int) -> np.ndarray: ...
# img: HxWx3 uint8. Returns same shape, values quantised to 2^bits levels.

def median_filter_squeeze(img: np.ndarray, kernel: int) -> np.ndarray: ...
# kernel: odd int (3, 5, or 7). Returns same shape after scipy median filter.
```

### `attack.py`
```python
def apply_attack(img: np.ndarray, faces, epsilon: float = 0.15) -> np.ndarray: ...
# faces: insightface detection result. Returns img with glasses noise overlaid.
```

### `detector.py`
```python
def get_embedding(img: np.ndarray) -> np.ndarray: ...
# Returns 512-dim L2-normalised ArcFace embedding.

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float: ...

def detect(embed_original, embed_bit, embed_median, threshold=0.50) -> tuple[bool, float]: ...
# Returns (is_adversarial, shift_score)
```

### `dataset.py`
```python
def load_dataset(path: str) -> dict[str, np.ndarray]: ...
# Returns {name: embedding} for all images in dataset/

def register_identity(name: str, img1: np.ndarray, img2: np.ndarray, state: dict) -> dict: ...
# Embeds both photos, averages + L2-normalises, adds to session state. Returns updated state.
```

---

## 8. Deployment

1. Create GitHub repo `feature-squeezing-demo`
2. Create HF Space linked to the repo (Gradio SDK, Python 3.10)
3. Add `README.md` HF Spaces header:
   ```yaml
   ---
   title: Feature Squeezing Demo
   emoji: 🛡
   colorFrom: indigo
   colorTo: purple
   sdk: gradio
   sdk_version: "4.x"
   app_file: app.py
   pinned: false
   ---
   ```
4. Push — Space auto-builds, model weights download on first boot (~2 min)
5. Share public URL with class

---

## 9. Dataset

- 3 pre-loaded identities, 2 photos each (6 images total)
- Source: LFW (Labeled Faces in the Wild) subset — public domain
- Stored as JPEGs in `dataset/` folder in the repo
- Selection criteria: clear frontal face, good lighting, varied demographics
- Live registration: 1 additional identity can be added during the session (2 photo uploads required); session state only, not persisted

---

## 10. Threshold rationale

The demo surfaces **two distinct thresholds** with different origins:

### Identity-match threshold — 0.72 (bar chart)

The bar chart in the Pipeline Demo tab now shows **0.72** as the primary
"recognised / rejected" line. This value comes from the ArcFace paper
(Deng et al., 2019) calibrated on LFW: at cosine similarity ≥ 0.72 the
model declares the same identity with ~99.8 % accuracy.

A secondary faint line at **0.50** is retained as a "demo default" reference.
It was a convenient round number used during development but is *not*
ROC-calibrated; it is shown only so viewers understand the original threshold
displayed in earlier versions of the demo.

### Self-shift detection threshold — 0.50 (SqueezeDetector)

`SqueezeDetector.detect()` computes the *self-shift*: how much the embedding
of the squeezed image drifts from the embedding of the attacked image.
This quantity is conceptually different from identity-match similarity —
it measures whether squeezing perturbs the embedding more than a clean image
would expect. The Xu et al. (NDSS 2018) paper uses an empirically chosen
threshold; 0.50 is the value inherited from the per-step plan and is left
unchanged here.

**The defense-verdict panel in the UI explicitly labels these as separate
quantities so the audience is not confused.**

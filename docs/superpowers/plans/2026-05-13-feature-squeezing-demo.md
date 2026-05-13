# Feature Squeezing Live Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Hugging Face Spaces Gradio app that demonstrates the Feature Squeezing defense defeating simulated adversarial glasses attacks on ArcFace face recognition, with a Bold & Modern dark UI.

**Architecture:** Five OOP modules (`squeezers.py`, `attack.py`, `detector.py`, `dataset.py`, `app.py`) wired by dependency injection. Business logic is fully unit-tested with mocked insightface. Gradio Blocks renders a two-tab UI: Pipeline Demo and Verify Identity. Matplotlib figures render all visualisations styled to the dark theme.

**Tech Stack:** Python 3.10, Gradio 4.x, insightface (ArcFace buffalo_l), onnxruntime, numpy, scipy, Pillow, matplotlib, pytest, pytest-mock

---

## OOP Contracts (read before touching any code)

```
BaseSqueezer (ABC)
  └── BitDepthSqueezer(bits: int)
  └── MedianFilterSqueezer(kernel: int)

Face (dataclass)
  └── bbox: np.ndarray        # [x1, y1, x2, y2]
  └── kps: np.ndarray         # 5×2 keypoints: left_eye, right_eye, nose, mouth_l, mouth_r

GlassesAttacker(epsilon: float)
  └── apply(img, face) -> np.ndarray
  └── _make_glasses_mask(shape, face) -> np.ndarray  [private]

FaceDetector()
  └── detect(img) -> list[Face]

ArcFaceEmbedder(detector: FaceDetector)
  └── embed(img) -> np.ndarray | None   # 512-dim L2-normed, None if no face found

Identity (dataclass)
  └── name: str
  └── embedding: np.ndarray
  └── photo: np.ndarray

IdentityDatabase(embedder: ArcFaceEmbedder, dataset_path: str)
  └── load() -> None
  └── register(name, photo1, photo2) -> Identity
  └── get(name) -> Identity
  └── names() -> list[str]

SqueezeDetector(embedder, squeezers: list[BaseSqueezer], threshold: float = 0.50)
  └── detect(target_embed, original_img, attacked_img) -> DetectionResult

DetectionResult (dataclass)
  └── is_adversarial: bool
  └── max_shift: float
  └── sims: dict[str, float]          # keys: "original","attacked","bit","median"
  └── squeezed_imgs: dict[str, np.ndarray]  # keys: "bit","median"
```

---

## File Map

| File | Responsibility |
|---|---|
| `squeezers.py` | `BaseSqueezer` ABC + `BitDepthSqueezer` + `MedianFilterSqueezer` |
| `attack.py` | `Face` dataclass + `GlassesAttacker` |
| `detector.py` | `FaceDetector` + `ArcFaceEmbedder` + `DetectionResult` + `SqueezeDetector` |
| `dataset.py` | `Identity` dataclass + `IdentityDatabase` |
| `app.py` | Gradio Blocks UI — CSS, Tab 1, Tab 2, event wiring, service bootstrap |
| `tests/conftest.py` | Shared fixtures: dummy images, mock embedder, mock detector |
| `tests/test_squeezers.py` | Unit tests for both squeezers |
| `tests/test_attack.py` | Unit tests for `GlassesAttacker` |
| `tests/test_detector.py` | Unit tests for `SqueezeDetector` (mocked embedder) |
| `tests/test_dataset.py` | Unit tests for `IdentityDatabase` (mocked embedder) |
| `dataset/` | 6 JPEG images: `alice_1.jpg`, `alice_2.jpg`, `bob_1.jpg`, `bob_2.jpg`, `carol_1.jpg`, `carol_2.jpg` |
| `requirements.txt` | All dependencies pinned |
| `README.md` | HF Spaces YAML header + project description |

---

## Task 1 — Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `tests/conftest.py`
- Create: `.gitignore`

- [ ] **Step 1: Create project directory and requirements.txt**

```
gradio==4.44.0
insightface==0.7.3
onnxruntime==1.18.0
numpy==1.26.4
scipy==1.13.1
Pillow==10.3.0
matplotlib==3.9.0
pytest==8.2.0
pytest-mock==3.14.0
```

- [ ] **Step 2: Create README.md with HF Spaces header**

```markdown
---
title: Feature Squeezing Demo
emoji: 🛡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# Feature Squeezing: Defeating Adversarial Glasses

Interactive demo for HUP Seminar. Demonstrates how Feature Squeezing (Xu et al., NDSS 2018)
detects adversarial examples crafted via simulated physical glasses attacks (Sharif et al., CCS 2016).
```

- [ ] **Step 3: Create tests/conftest.py with shared fixtures**

```python
import numpy as np
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def blank_face_img() -> np.ndarray:
    """160×160 RGB image filled with a mid-grey face-like gradient."""
    img = np.full((160, 160, 3), 128, dtype=np.uint8)
    # bright circle to simulate face region
    cx, cy, r = 80, 80, 60
    y, x = np.ogrid[:160, :160]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    img[mask] = [200, 170, 150]
    return img


@pytest.fixture
def mock_face():
    """Minimal Face-like object with eye keypoints at expected positions."""
    from attack import Face
    kps = np.array([
        [55.0, 65.0],   # left eye
        [105.0, 65.0],  # right eye
        [80.0, 90.0],   # nose
        [60.0, 110.0],  # mouth left
        [100.0, 110.0], # mouth right
    ], dtype=np.float32)
    bbox = np.array([20.0, 30.0, 140.0, 145.0], dtype=np.float32)
    return Face(bbox=bbox, kps=kps)


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = np.random.randn(512).astype(np.float32)
    # L2-normalise so it behaves like a real embedding
    v = embedder.embed.return_value
    embedder.embed.return_value = v / np.linalg.norm(v)
    return embedder
```

- [ ] **Step 4: Create .gitignore**

```
__pycache__/
*.pyc
.pytest_cache/
*.onnx
/root/.insightface/
.insightface/
*.egg-info/
dist/
.env
```

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt README.md tests/conftest.py .gitignore
git commit -m "chore: project scaffold"
```

---

## Task 2 — BitDepthSqueezer

**Files:**
- Create: `squeezers.py`
- Create: `tests/test_squeezers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_squeezers.py
import numpy as np
import pytest
from squeezers import BitDepthSqueezer


def test_bit_depth_squeezer_reduces_unique_values():
    img = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
    img = np.repeat(img, 3, axis=2)
    squeezer = BitDepthSqueezer(bits=2)
    result = squeezer.squeeze(img)
    assert result.dtype == np.uint8
    unique_vals = np.unique(result)
    assert len(unique_vals) <= 4   # 2^2 = 4 levels


def test_bit_depth_squeezer_preserves_shape(blank_face_img):
    squeezer = BitDepthSqueezer(bits=4)
    result = squeezer.squeeze(blank_face_img)
    assert result.shape == blank_face_img.shape


def test_bit_depth_squeezer_1_bit_gives_two_values():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = BitDepthSqueezer(bits=1).squeeze(img)
    assert set(np.unique(result)).issubset({0, 128})


def test_bit_depth_squeezer_8_bits_is_identity():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = BitDepthSqueezer(bits=8).squeeze(img)
    np.testing.assert_array_equal(result, img)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_squeezers.py -v
```
Expected: `ImportError: No module named 'squeezers'`

- [ ] **Step 3: Implement squeezers.py with BaseSqueezer and BitDepthSqueezer**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseSqueezer(ABC):
    @abstractmethod
    def squeeze(self, img: np.ndarray) -> np.ndarray:
        """Apply squeezing transform. Input and output are HxWx3 uint8."""


class BitDepthSqueezer(BaseSqueezer):
    def __init__(self, bits: int = 4) -> None:
        if not 1 <= bits <= 8:
            raise ValueError(f"bits must be between 1 and 8, got {bits}")
        self.bits = bits

    def squeeze(self, img: np.ndarray) -> np.ndarray:
        levels = 2 ** self.bits
        if levels == 256:
            return img.copy()
        step = 256 // levels
        quantised = (img.astype(np.uint16) // step) * step
        return np.clip(quantised, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_squeezers.py -v -k "bit_depth"
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add squeezers.py tests/test_squeezers.py
git commit -m "feat: BitDepthSqueezer with BaseSqueezer ABC"
```

---

## Task 3 — MedianFilterSqueezer

**Files:**
- Modify: `squeezers.py`
- Modify: `tests/test_squeezers.py`

- [ ] **Step 1: Add failing tests for MedianFilterSqueezer**

```python
# append to tests/test_squeezers.py
from squeezers import MedianFilterSqueezer


def test_median_squeezer_removes_salt_and_pepper(blank_face_img):
    noisy = blank_face_img.copy()
    # inject isolated noise pixels
    rng = np.random.default_rng(42)
    coords = rng.integers(0, 160, size=(200, 2))
    noisy[coords[:, 0], coords[:, 1]] = 255
    squeezer = MedianFilterSqueezer(kernel=3)
    result = squeezer.squeeze(noisy)
    # result should be closer to original than noisy is
    diff_noisy = np.mean(np.abs(noisy.astype(int) - blank_face_img.astype(int)))
    diff_result = np.mean(np.abs(result.astype(int) - blank_face_img.astype(int)))
    assert diff_result < diff_noisy


def test_median_squeezer_preserves_shape(blank_face_img):
    result = MedianFilterSqueezer(kernel=5).squeeze(blank_face_img)
    assert result.shape == blank_face_img.shape
    assert result.dtype == np.uint8


def test_median_squeezer_rejects_even_kernel():
    with pytest.raises(ValueError, match="odd"):
        MedianFilterSqueezer(kernel=4)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_squeezers.py -v -k "median"
```
Expected: `ImportError` or `NameError`

- [ ] **Step 3: Add MedianFilterSqueezer to squeezers.py**

```python
# append to squeezers.py
from scipy.ndimage import median_filter


class MedianFilterSqueezer(BaseSqueezer):
    def __init__(self, kernel: int = 3) -> None:
        if kernel % 2 == 0:
            raise ValueError(f"kernel must be odd, got {kernel}")
        self.kernel = kernel

    def squeeze(self, img: np.ndarray) -> np.ndarray:
        result = np.stack([
            median_filter(img[:, :, c], size=self.kernel)
            for c in range(img.shape[2])
        ], axis=2)
        return result.astype(np.uint8)
```

- [ ] **Step 4: Run all squeezer tests**

```bash
pytest tests/test_squeezers.py -v
```
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add squeezers.py tests/test_squeezers.py
git commit -m "feat: MedianFilterSqueezer"
```

---

## Task 4 — GlassesAttacker

**Files:**
- Create: `attack.py`
- Create: `tests/test_attack.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_attack.py
import numpy as np
import pytest
from attack import Face, GlassesAttacker


def test_apply_changes_eye_region(blank_face_img, mock_face):
    attacker = GlassesAttacker(epsilon=0.15)
    result = attacker.apply(blank_face_img, mock_face)
    # eye region should differ from original
    y1, y2 = 50, 80   # approximate eye-band rows for mock_face kps
    eye_region_orig = blank_face_img[y1:y2, 40:120]
    eye_region_result = result[y1:y2, 40:120]
    assert not np.array_equal(eye_region_orig, eye_region_result)


def test_apply_preserves_shape_and_dtype(blank_face_img, mock_face):
    result = GlassesAttacker(epsilon=0.10).apply(blank_face_img, mock_face)
    assert result.shape == blank_face_img.shape
    assert result.dtype == np.uint8


def test_apply_stays_in_valid_pixel_range(blank_face_img, mock_face):
    result = GlassesAttacker(epsilon=0.99).apply(blank_face_img, mock_face)
    assert result.min() >= 0
    assert result.max() <= 255


def test_make_glasses_mask_covers_eye_region(mock_face):
    attacker = GlassesAttacker()
    mask = attacker._make_glasses_mask((160, 160, 3), mock_face)
    assert mask.shape == (160, 160, 1)
    # mask should be 1 around eye y-region
    eye_y = int(mock_face.kps[:2, 1].mean())
    assert mask[eye_y, 80, 0] == 1.0
    # mask should be 0 far below face
    assert mask[155, 80, 0] == 0.0
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_attack.py -v
```
Expected: `ImportError: No module named 'attack'`

- [ ] **Step 3: Implement attack.py**

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Face:
    bbox: np.ndarray   # shape (4,): [x1, y1, x2, y2]
    kps: np.ndarray    # shape (5, 2): left_eye, right_eye, nose, mouth_l, mouth_r


class GlassesAttacker:
    def __init__(self, epsilon: float = 0.15) -> None:
        if not 0.0 < epsilon <= 1.0:
            raise ValueError(f"epsilon must be in (0, 1], got {epsilon}")
        self.epsilon = epsilon

    def apply(self, img: np.ndarray, face: Face) -> np.ndarray:
        mask = self._make_glasses_mask(img.shape, face)
        noise = np.random.uniform(
            -self.epsilon * 255,
            self.epsilon * 255,
            img.shape,
        ).astype(np.float32)
        attacked = img.astype(np.float32) + noise * mask
        return np.clip(attacked, 0, 255).astype(np.uint8)

    def _make_glasses_mask(
        self, img_shape: tuple[int, ...], face: Face
    ) -> np.ndarray:
        left_eye, right_eye = face.kps[0], face.kps[1]
        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        cx = (left_eye[0] + right_eye[0]) / 2
        cy = (left_eye[1] + right_eye[1]) / 2

        x1 = max(0, int(left_eye[0] - eye_dist * 0.35))
        x2 = min(img_shape[1], int(right_eye[0] + eye_dist * 0.35))
        y1 = max(0, int(cy - eye_dist * 0.32))
        y2 = min(img_shape[0], int(cy + eye_dist * 0.38))

        mask = np.zeros(img_shape[:2], dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        return mask[:, :, np.newaxis]  # HxWx1 for broadcasting
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_attack.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add attack.py tests/test_attack.py
git commit -m "feat: GlassesAttacker with Face dataclass"
```

---

## Task 5 — FaceDetector and ArcFaceEmbedder

**Files:**
- Create: `detector.py`
- Create: `tests/test_detector.py` (partial — embedder tests using mocks)

- [ ] **Step 1: Write failing tests for ArcFaceEmbedder using a mocked FaceDetector**

```python
# tests/test_detector.py
import numpy as np
import pytest
from unittest.mock import MagicMock
from detector import ArcFaceEmbedder, FaceDetector


def _make_mock_insightface_face(embedding: np.ndarray):
    face = MagicMock()
    face.normed_embedding = embedding
    return face


def test_embed_returns_512_dim_vector(blank_face_img):
    raw_embed = np.random.randn(512).astype(np.float32)
    raw_embed /= np.linalg.norm(raw_embed)

    mock_detector = MagicMock(spec=FaceDetector)
    mock_detector.detect.return_value = [_make_mock_insightface_face(raw_embed)]

    embedder = ArcFaceEmbedder(detector=mock_detector)
    result = embedder.embed(blank_face_img)

    assert result is not None
    assert result.shape == (512,)
    assert result.dtype == np.float32


def test_embed_returns_none_when_no_face_detected(blank_face_img):
    mock_detector = MagicMock(spec=FaceDetector)
    mock_detector.detect.return_value = []

    embedder = ArcFaceEmbedder(detector=mock_detector)
    assert embedder.embed(blank_face_img) is None


def test_embed_uses_first_face_when_multiple_detected(blank_face_img):
    embed_a = np.ones(512, dtype=np.float32); embed_a /= np.linalg.norm(embed_a)
    embed_b = np.zeros(512, dtype=np.float32); embed_b[0] = 1.0

    mock_detector = MagicMock(spec=FaceDetector)
    mock_detector.detect.return_value = [
        _make_mock_insightface_face(embed_a),
        _make_mock_insightface_face(embed_b),
    ]
    embedder = ArcFaceEmbedder(detector=mock_detector)
    result = embedder.embed(blank_face_img)
    np.testing.assert_array_equal(result, embed_a)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_detector.py -v
```
Expected: `ImportError: No module named 'detector'`

- [ ] **Step 3: Implement detector.py (FaceDetector + ArcFaceEmbedder)**

```python
from __future__ import annotations
import numpy as np


class FaceDetector:
    """Wraps insightface FaceAnalysis for face detection and landmark extraction."""

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis
        self._app = FaceAnalysis(providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, img: np.ndarray) -> list:
        """Return list of insightface Face objects detected in img (RGB uint8)."""
        return self._app.get(img)


class ArcFaceEmbedder:
    """Extracts L2-normalised 512-dim ArcFace embeddings via a FaceDetector."""

    def __init__(self, detector: FaceDetector) -> None:
        self._detector = detector

    def embed(self, img: np.ndarray) -> np.ndarray | None:
        """Return 512-dim embedding for the first detected face, or None."""
        faces = self._detector.detect(img)
        if not faces:
            return None
        return faces[0].normed_embedding.astype(np.float32)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_detector.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add detector.py tests/test_detector.py
git commit -m "feat: FaceDetector and ArcFaceEmbedder"
```

---

## Task 6 — DetectionResult and SqueezeDetector

**Files:**
- Modify: `detector.py`
- Modify: `tests/test_detector.py`

- [ ] **Step 1: Add failing tests for SqueezeDetector**

```python
# append to tests/test_detector.py
import numpy as np
from unittest.mock import MagicMock
from detector import SqueezeDetector, ArcFaceEmbedder
from squeezers import BitDepthSqueezer, MedianFilterSqueezer


def _normed(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_squeeze_detector_flags_adversarial_when_shift_exceeds_threshold(blank_face_img):
    target_embed = _normed(np.ones(512))
    original_embed = _normed(np.ones(512))          # sim ≈ 1.0
    attacked_embed = _normed(-np.ones(512))          # sim ≈ -1.0 (opposite)
    # squeezed embeddings close to target — shift is large
    squeezed_embed = _normed(np.ones(512) * 0.9)

    mock_embedder = MagicMock(spec=ArcFaceEmbedder)
    mock_embedder.embed.side_effect = [
        original_embed,    # original call
        attacked_embed,    # attacked call
        squeezed_embed,    # bit squeezer call
        squeezed_embed,    # median squeezer call
    ]

    bit_sq = MagicMock(); bit_sq.squeeze.return_value = blank_face_img
    med_sq = MagicMock(); med_sq.squeeze.return_value = blank_face_img

    detector = SqueezeDetector(
        embedder=mock_embedder,
        squeezers=[bit_sq, med_sq],
        threshold=0.50,
    )
    result = detector.detect(target_embed, blank_face_img, blank_face_img)
    assert result.is_adversarial is True
    assert result.max_shift > 0.50


def test_squeeze_detector_clean_input_not_flagged(blank_face_img):
    target_embed = _normed(np.ones(512))
    similar_embed = _normed(np.ones(512) * 0.98)

    mock_embedder = MagicMock(spec=ArcFaceEmbedder)
    mock_embedder.embed.return_value = similar_embed

    bit_sq = MagicMock(); bit_sq.squeeze.return_value = blank_face_img
    med_sq = MagicMock(); med_sq.squeeze.return_value = blank_face_img

    detector = SqueezeDetector(
        embedder=mock_embedder,
        squeezers=[bit_sq, med_sq],
        threshold=0.50,
    )
    result = detector.detect(target_embed, blank_face_img, blank_face_img)
    assert result.is_adversarial is False


def test_detection_result_contains_all_sim_keys(blank_face_img):
    embed = _normed(np.random.randn(512))
    mock_embedder = MagicMock(spec=ArcFaceEmbedder)
    mock_embedder.embed.return_value = embed

    bit_sq = MagicMock(); bit_sq.squeeze.return_value = blank_face_img
    med_sq = MagicMock(); med_sq.squeeze.return_value = blank_face_img

    detector = SqueezeDetector(mock_embedder, [bit_sq, med_sq])
    result = detector.detect(embed, blank_face_img, blank_face_img)
    assert set(result.sims.keys()) == {"original", "attacked", "bit", "median"}
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_detector.py -v -k "squeeze_detector"
```
Expected: `ImportError` on `SqueezeDetector`

- [ ] **Step 3: Add DetectionResult dataclass and SqueezeDetector to detector.py**

```python
# append to detector.py
from dataclasses import dataclass, field
from squeezers import BaseSqueezer


@dataclass
class DetectionResult:
    is_adversarial: bool
    max_shift: float
    sims: dict[str, float]
    squeezed_imgs: dict[str, np.ndarray]


class SqueezeDetector:
    """Detects adversarial inputs by measuring ArcFace embedding shift after squeezing."""

    _SQUEEZER_KEYS = ("bit", "median")

    def __init__(
        self,
        embedder: ArcFaceEmbedder,
        squeezers: list[BaseSqueezer],
        threshold: float = 0.50,
    ) -> None:
        if len(squeezers) != 2:
            raise ValueError("Exactly 2 squeezers required (bit, median)")
        self._embedder = embedder
        self._squeezers = squeezers
        self._threshold = threshold

    def detect(
        self,
        target_embed: np.ndarray,
        original_img: np.ndarray,
        attacked_img: np.ndarray,
    ) -> DetectionResult:
        embed_original = self._embedder.embed(original_img) or target_embed
        embed_attacked = self._embedder.embed(attacked_img) or target_embed

        squeezed_imgs = {
            key: sq.squeeze(attacked_img)
            for key, sq in zip(self._SQUEEZER_KEYS, self._squeezers)
        }
        squeezed_embeds = {
            key: self._embedder.embed(img) or target_embed
            for key, img in squeezed_imgs.items()
        }

        sims = {
            "original": float(np.dot(target_embed, embed_original)),
            "attacked": float(np.dot(target_embed, embed_attacked)),
            "bit":      float(np.dot(target_embed, squeezed_embeds["bit"])),
            "median":   float(np.dot(target_embed, squeezed_embeds["median"])),
        }

        sim_attacked = sims["attacked"]
        shifts = [abs(sims[k] - sim_attacked) for k in ("bit", "median")]
        max_shift = max(shifts)

        return DetectionResult(
            is_adversarial=max_shift > self._threshold,
            max_shift=max_shift,
            sims=sims,
            squeezed_imgs=squeezed_imgs,
        )
```

- [ ] **Step 4: Run all detector tests**

```bash
pytest tests/test_detector.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add detector.py tests/test_detector.py
git commit -m "feat: SqueezeDetector and DetectionResult"
```

---

## Task 7 — IdentityDatabase

**Files:**
- Create: `dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dataset.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from dataset import Identity, IdentityDatabase
from detector import ArcFaceEmbedder


def _normed(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def mock_embedder_with_known_embed():
    embed = _normed(np.random.randn(512))
    m = MagicMock(spec=ArcFaceEmbedder)
    m.embed.return_value = embed
    return m, embed


def test_register_returns_identity_with_averaged_embedding(
    blank_face_img, mock_embedder_with_known_embed
):
    embedder, embed = mock_embedder_with_known_embed
    db = IdentityDatabase(embedder=embedder, dataset_path="dataset")
    identity = db.register("alice", blank_face_img, blank_face_img)

    assert isinstance(identity, Identity)
    assert identity.name == "alice"
    assert identity.embedding.shape == (512,)
    # embedding should be L2-normalised
    norm = np.linalg.norm(identity.embedding)
    assert abs(norm - 1.0) < 1e-5


def test_register_adds_to_names(blank_face_img, mock_embedder_with_known_embed):
    embedder, _ = mock_embedder_with_known_embed
    db = IdentityDatabase(embedder=embedder, dataset_path="dataset")
    db.register("alice", blank_face_img, blank_face_img)
    assert "alice" in db.names()


def test_get_returns_registered_identity(blank_face_img, mock_embedder_with_known_embed):
    embedder, _ = mock_embedder_with_known_embed
    db = IdentityDatabase(embedder=embedder, dataset_path="dataset")
    db.register("bob", blank_face_img, blank_face_img)
    identity = db.get("bob")
    assert identity.name == "bob"


def test_get_unknown_name_raises_key_error(mock_embedder_with_known_embed):
    embedder, _ = mock_embedder_with_known_embed
    db = IdentityDatabase(embedder=embedder, dataset_path="dataset")
    with pytest.raises(KeyError):
        db.get("nobody")
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_dataset.py -v
```
Expected: `ImportError: No module named 'dataset'`

- [ ] **Step 3: Implement dataset.py**

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from detector import ArcFaceEmbedder


@dataclass
class Identity:
    name: str
    embedding: np.ndarray   # 512-dim, L2-normalised average of 2 photos
    photo: np.ndarray       # first photo as representative image (RGB uint8)


class IdentityDatabase:
    """Loads pre-built identities from disk and supports one live registration."""

    def __init__(self, embedder: ArcFaceEmbedder, dataset_path: str) -> None:
        self._embedder = embedder
        self._path = Path(dataset_path)
        self._identities: dict[str, Identity] = {}

    def load(self) -> None:
        """Load all {name}_1.jpg / {name}_2.jpg pairs from dataset_path."""
        names = sorted({
            f.stem.replace("_1", "")
            for f in self._path.glob("*_1.jpg")
        })
        for name in names:
            img1 = self._load_img(self._path / f"{name}_1.jpg")
            img2 = self._load_img(self._path / f"{name}_2.jpg")
            self.register(name, img1, img2)

    def register(
        self, name: str, photo1: np.ndarray, photo2: np.ndarray
    ) -> Identity:
        """Embed two photos, average + L2-normalise, store and return Identity."""
        e1 = self._embedder.embed(photo1)
        e2 = self._embedder.embed(photo2)
        if e1 is None or e2 is None:
            raise ValueError(f"Could not detect a face in photos for '{name}'")
        avg = (e1 + e2) / 2.0
        avg = (avg / np.linalg.norm(avg)).astype(np.float32)
        identity = Identity(name=name, embedding=avg, photo=photo1)
        self._identities[name] = identity
        return identity

    def get(self, name: str) -> Identity:
        return self._identities[name]

    def names(self) -> list[str]:
        return list(self._identities.keys())

    @staticmethod
    def _load_img(path: Path) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"))
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_dataset.py -v
```
Expected: 4 passed

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
pytest -v
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add dataset.py tests/test_dataset.py
git commit -m "feat: Identity dataclass and IdentityDatabase"
```

---

## Task 8 — Dataset Preparation

**Files:**
- Create: `dataset/alice_1.jpg`, `dataset/alice_2.jpg`
- Create: `dataset/bob_1.jpg`, `dataset/bob_2.jpg`
- Create: `dataset/carol_1.jpg`, `dataset/carol_2.jpg`
- Create: `dataset/prepare_dataset.py` (one-time helper script)

- [ ] **Step 1: Create the dataset helper script**

```python
# dataset/prepare_dataset.py
# Run once to download 6 LFW images. Not part of the app.
import urllib.request
from pathlib import Path

# LFW image URLs (public domain). Substitute with your chosen identities.
# Format: (save_name, url)
IMAGES = [
    ("alice_1", "https://vis-www.cs.umass.edu/lfw/images/George_W_Bush/George_W_Bush_0001.jpg"),
    ("alice_2", "https://vis-www.cs.umass.edu/lfw/images/George_W_Bush/George_W_Bush_0002.jpg"),
    ("bob_1",   "https://vis-www.cs.umass.edu/lfw/images/Colin_Powell/Colin_Powell_0001.jpg"),
    ("bob_2",   "https://vis-www.cs.umass.edu/lfw/images/Colin_Powell/Colin_Powell_0002.jpg"),
    ("carol_1", "https://vis-www.cs.umass.edu/lfw/images/Tony_Blair/Tony_Blair_0001.jpg"),
    ("carol_2", "https://vis-www.cs.umass.edu/lfw/images/Tony_Blair/Tony_Blair_0002.jpg"),
]

out = Path(__file__).parent
for name, url in IMAGES:
    dest = out / f"{name}.jpg"
    if not dest.exists():
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  saved → {dest}")
print("Done.")
```

- [ ] **Step 2: Run the script to download images**

```bash
python dataset/prepare_dataset.py
```
Expected: 6 JPEG files appear in `dataset/`

- [ ] **Step 3: Visually verify images**

Open each file and confirm: clear frontal face, no sunglasses, good lighting. Replace any failed downloads with a different LFW identity pair using the same `{name}_1.jpg` / `{name}_2.jpg` naming convention.

- [ ] **Step 4: Verify insightface can detect faces in all 6 images**

```python
# run in python REPL or a scratch cell
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
from pathlib import Path

app = FaceAnalysis(providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

for p in sorted(Path("dataset").glob("*.jpg")):
    img = np.array(Image.open(p).convert("RGB"))
    faces = app.get(img)
    print(f"{p.name}: {len(faces)} face(s) detected")
```
Expected: each file prints `1 face(s) detected`

- [ ] **Step 5: Commit dataset images**

```bash
git add dataset/*.jpg
git commit -m "data: add 3 LFW identities (6 images) for demo dataset"
```

---

## Task 9 — Gradio App: CSS Theme and Service Bootstrap

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create app.py with the dark theme CSS and service bootstrap**

```python
# app.py
from __future__ import annotations
import numpy as np
import gradio as gr
from detector import FaceDetector, ArcFaceEmbedder, SqueezeDetector
from squeezers import BitDepthSqueezer, MedianFilterSqueezer
from dataset import IdentityDatabase

# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
:root {
  --bg-base:     #0f0f1a;
  --bg-surface:  #1a1a2e;
  --bg-card:     rgba(255,255,255,0.04);
  --border-dim:  rgba(255,255,255,0.08);
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
  background: var(--bg-base) !important;
  font-family: 'Inter', system-ui, sans-serif;
}
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
.gr-panel, .gr-box { background: var(--bg-surface) !important; border: 1px solid var(--border-dim) !important; border-radius: 12px !important; }
label, .gr-label { color: var(--text-muted) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
.gr-button { border-radius: 8px !important; font-weight: 600 !important; }
.btn-attack { background: var(--red-dim) !important; border: 1px solid var(--red) !important; color: #fca5a5 !important; }
.btn-defend { background: var(--green-dim) !important; border: 1px solid var(--green) !important; color: #6ee7b7 !important; }
.btn-register { background: var(--purple-dim) !important; border: 1px solid var(--purple) !important; color: #c4b5fd !important; }
.squeezer-card { background: rgba(167,139,250,0.06) !important; border: 1px solid rgba(167,139,250,0.25) !important; border-radius: 14px !important; padding: 20px !important; }
.gr-slider input[type=range]::-webkit-slider-thumb { background: var(--purple) !important; }
"""

# ── Service bootstrap ────────────────────────────────────────────────────────

def _build_services():
    detector = FaceDetector()
    embedder = ArcFaceEmbedder(detector)
    squeezers = [BitDepthSqueezer(bits=4), MedianFilterSqueezer(kernel=3)]
    squeeze_detector = SqueezeDetector(embedder=embedder, squeezers=squeezers)
    db = IdentityDatabase(embedder=embedder, dataset_path="dataset")
    db.load()
    return embedder, squeeze_detector, db


EMBEDDER, SQUEEZE_DETECTOR, DB = _build_services()
```

- [ ] **Step 2: Verify the app bootstraps without error**

```bash
python -c "import app; print('Bootstrap OK — identities:', app.DB.names())"
```
Expected: `Bootstrap OK — identities: ['alice', 'bob', 'carol']`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: app bootstrap — services, CSS theme"
```

---

## Task 10 — Tab 1: Pipeline Demo

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the matplotlib pipeline figure helper**

```python
# add to app.py, after service bootstrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image as PILImage


_STAGE_COLORS = {
    "original": "#94a3b8",
    "attacked": "#ef4444",
    "bit":      "#fbbf24",
    "median":   "#10b981",
}
_STAGE_LABELS = {
    "original": "Original",
    "attacked": "Adv. Glasses",
    "bit":      "Bit Squeezed",
    "median":   "Median Filter",
}


def _render_pipeline_figure(
    imgs: dict[str, np.ndarray],
    sims: dict[str, float],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.patch.set_facecolor("#0f0f1a")

    for ax, key in zip(axes, ["original", "attacked", "bit", "median"]):
        color = _STAGE_COLORS[key]
        img = imgs.get(key)
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor("#1a1a2e")

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(_STAGE_LABELS[key], color=color, fontsize=11, pad=8)

        sim = sims.get(key)
        if sim is not None:
            ax.set_xlabel(f"sim: {sim:.2f}", color=color, fontsize=13,
                          fontweight="bold", labelpad=6)

    fig.tight_layout(pad=1.5)
    return fig


def _render_similarity_bars(sims: dict[str, float]) -> plt.Figure:
    keys   = ["original", "attacked", "bit", "median"]
    labels = [_STAGE_LABELS[k] for k in keys]
    values = [sims.get(k, 0.0) for k in keys]
    colors = [_STAGE_COLORS[k] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 2.6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    y_pos = range(len(keys))
    bars = ax.barh(list(y_pos), values, color=colors, height=0.5, alpha=0.85)
    ax.axvline(x=0.50, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlim(0, 1.05)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, color="#94a3b8", fontsize=11)
    ax.tick_params(axis="x", colors="#475569")
    ax.text(0.51, -0.7, "threshold: 0.50", color="#ef4444", fontsize=9, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, val, color in zip(bars, values, colors):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", color=color, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
```

- [ ] **Step 2: Add Tab 1 event handlers**

```python
# add to app.py

def _img_array(pil_or_array) -> np.ndarray | None:
    if pil_or_array is None:
        return None
    if isinstance(pil_or_array, np.ndarray):
        return pil_or_array
    return np.array(pil_or_array)


def on_identity_select(name: str):
    identity = DB.get(name)
    empty_sims = {k: 0.0 for k in ["original", "attacked", "bit", "median"]}
    empty_imgs = {k: None for k in ["original", "attacked", "bit", "median"]}
    empty_imgs["original"] = identity.photo
    pipeline_fig = _render_pipeline_figure(empty_imgs, empty_sims)
    sim_fig = _render_similarity_bars(empty_sims)
    return (
        identity.photo,          # target_photo display
        identity.embedding,      # gr.State: target_embed
        identity.photo,          # gr.State: original_img
        None,                    # gr.State: attacked_img
        pipeline_fig,
        sim_fig,
        gr.update(visible=False),  # verdict row hidden until detect runs
    )


def on_attack(original_img, target_embed, bits, kernel):
    if original_img is None:
        return gr.update(), gr.update(), None, "Select an identity first."
    from attack import GlassesAttacker
    from insightface.app import FaceAnalysis
    app_if = FaceAnalysis(providers=["CPUExecutionProvider"])
    app_if.prepare(ctx_id=0, det_size=(640, 640))

    from attack import Face
    faces_raw = app_if.get(original_img)
    if not faces_raw:
        return gr.update(), gr.update(), None, "No face detected in image."

    f = faces_raw[0]
    face = Face(bbox=f.bbox, kps=f.kps)
    attacker = GlassesAttacker(epsilon=0.15)
    attacked = attacker.apply(original_img, face)

    empty_sims = {k: 0.0 for k in ["original", "attacked", "bit", "median"]}
    imgs = {"original": original_img, "attacked": attacked, "bit": None, "median": None}
    pipeline_fig = _render_pipeline_figure(imgs, empty_sims)
    sim_fig = _render_similarity_bars(empty_sims)
    return pipeline_fig, sim_fig, attacked, ""


def on_squeeze_detect(original_img, attacked_img, target_embed, bits, kernel):
    if attacked_img is None or target_embed is None:
        return gr.update(), gr.update(), gr.update(visible=False)

    from squeezers import BitDepthSqueezer, MedianFilterSqueezer
    from detector import SqueezeDetector

    squeezers = [BitDepthSqueezer(bits=int(bits)), MedianFilterSqueezer(kernel=int(kernel))]
    local_detector = SqueezeDetector(EMBEDDER, squeezers)
    result = local_detector.detect(target_embed, original_img, attacked_img)

    imgs = {
        "original": original_img,
        "attacked": attacked_img,
        "bit":      result.squeezed_imgs["bit"],
        "median":   result.squeezed_imgs["median"],
    }
    pipeline_fig = _render_pipeline_figure(imgs, result.sims)
    sim_fig = _render_similarity_bars(result.sims)

    if result.is_adversarial:
        verdict_html = (
            '<div style="background:rgba(239,68,68,0.15);border:1px solid #ef4444;'
            'border-radius:10px;padding:16px 24px;display:flex;align-items:center;gap:12px;">'
            '<span style="font-size:28px;">🚨</span>'
            '<div><div style="font-size:15px;color:#fca5a5;font-weight:bold;letter-spacing:1.5px;">'
            'ADVERSARIAL INPUT DETECTED</div>'
            f'<div style="font-size:11px;color:#64748b;margin-top:4px;">'
            f'Embedding shift: {result.max_shift:.3f} — above threshold (0.50). '
            'Attack neutralised by Feature Squeezing.</div></div></div>'
        )
    else:
        verdict_html = (
            '<div style="background:rgba(16,185,129,0.12);border:1px solid #10b981;'
            'border-radius:10px;padding:16px 24px;display:flex;align-items:center;gap:12px;">'
            '<span style="font-size:28px;">✅</span>'
            '<div><div style="font-size:15px;color:#6ee7b7;font-weight:bold;letter-spacing:1.5px;">'
            'CLEAN INPUT</div>'
            f'<div style="font-size:11px;color:#64748b;margin-top:4px;">'
            f'Embedding shift: {result.max_shift:.3f} — below threshold.</div></div></div>'
        )
    return pipeline_fig, sim_fig, gr.update(value=verdict_html, visible=True)


def on_register(name: str, photo1, photo2, db_state: dict):
    img1 = _img_array(photo1)
    img2 = _img_array(photo2)
    if img1 is None or img2 is None or not name.strip():
        return db_state, gr.update()
    DB.register(name.strip(), img1, img2)
    return db_state, gr.update(choices=DB.names(), value=name.strip())
```

- [ ] **Step 3: Build the Tab 1 Gradio layout and wire events**

```python
# add to app.py — the main Blocks definition

def build_app() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="Feature Squeezing Demo") as demo:
        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;padding:20px 0 8px;">
          <div style="font-size:28px;font-weight:800;color:#a78bfa;letter-spacing:2px;">
            🛡 FEATURE SQUEEZING
          </div>
          <div style="font-size:12px;color:#64748b;margin-top:4px;letter-spacing:1px;">
            Detecting Adversarial Examples in Deep Neural Networks
          </div>
          <div style="display:flex;gap:8px;justify-content:center;margin-top:10px;">
            <span style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
              border-radius:20px;padding:4px 14px;font-size:11px;color:#fca5a5;">
              ⚔ Accessorize to a Crime — Sharif et al., CCS 2016
            </span>
            <span style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
              border-radius:20px;padding:4px 14px;font-size:11px;color:#6ee7b7;">
              🛡 Feature Squeezing — Xu et al., NDSS 2018
            </span>
          </div>
        </div>
        """)

        # ── Session state ────────────────────────────────────────────────────
        state_target_embed = gr.State(value=None)
        state_original_img = gr.State(value=None)
        state_attacked_img = gr.State(value=None)
        state_db           = gr.State(value={})

        with gr.Tabs():
            # ── TAB 1 ────────────────────────────────────────────────────────
            with gr.Tab("⚔ Pipeline Demo"):
                with gr.Row():
                    # LEFT SIDEBAR
                    with gr.Column(scale=1, min_width=180):
                        identity_dd = gr.Dropdown(
                            choices=DB.names(),
                            label="Target Identity",
                            value=DB.names()[0],
                        )
                        target_photo = gr.Image(
                            label="Selected", height=120, interactive=False
                        )
                        gr.HTML('<hr style="border-color:rgba(255,255,255,0.06);margin:8px 0;">')
                        gr.HTML('<div style="font-size:9px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Add Your Face</div>')
                        reg_photo1 = gr.Image(label="Photo 1", height=80, type="numpy")
                        reg_photo2 = gr.Image(label="Photo 2", height=80, type="numpy")
                        reg_name   = gr.Textbox(label="Name", placeholder="Your name")
                        reg_btn    = gr.Button("＋ Register", elem_classes=["btn-register"])
                        gr.HTML('<hr style="border-color:rgba(255,255,255,0.06);margin:8px 0;">')
                        attack_btn = gr.Button("⚔ Launch Attack",   elem_classes=["btn-attack"])
                        defend_btn = gr.Button("🛡 Squeeze + Detect", elem_classes=["btn-defend"])

                    # MAIN PANEL
                    with gr.Column(scale=4):
                        # Squeezer controls hero card
                        with gr.Group(elem_classes=["squeezer-card"]):
                            gr.HTML('<div style="font-size:11px;color:#a78bfa;text-transform:uppercase;letter-spacing:2px;text-align:center;margin-bottom:12px;">⚙ Squeezer Controls</div>')
                            with gr.Row():
                                bits_slider   = gr.Slider(1, 8, value=4, step=1,
                                    label="Bit Depth  (1 = 2 levels … 8 = 256 levels)")
                                kernel_slider = gr.Slider(3, 7, value=3, step=2,
                                    label="Median Kernel  (3×3 / 5×5 / 7×7)")

                        pipeline_plot = gr.Plot(label="Pipeline", show_label=False)
                        sim_plot      = gr.Plot(label="Similarity vs Target", show_label=False)
                        verdict_html  = gr.HTML(visible=False)

                # ── Wire Tab 1 events ────────────────────────────────────────
                identity_dd.change(
                    fn=on_identity_select,
                    inputs=[identity_dd],
                    outputs=[target_photo, state_target_embed, state_original_img,
                             state_attacked_img, pipeline_plot, sim_plot, verdict_html],
                )
                attack_btn.click(
                    fn=on_attack,
                    inputs=[state_original_img, state_target_embed, bits_slider, kernel_slider],
                    outputs=[pipeline_plot, sim_plot, state_attacked_img, verdict_html],
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

        return demo


if __name__ == "__main__":
    build_app().launch()
```

- [ ] **Step 4: Run the app locally and test Tab 1 end-to-end**

```bash
python app.py
```
- Open `http://localhost:7860`
- Select an identity from the dropdown — target photo should appear
- Click "⚔ Launch Attack" — pipeline strip should show original + glasses image
- Adjust sliders — then click "🛡 Squeeze + Detect"
- Confirm: pipeline strip shows all 4 panels, similarity bars appear, verdict banner fires 🚨

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: Tab 1 pipeline demo — attack, squeeze, detect, verdict"
```

---

## Task 11 — Tab 2: Verify Identity

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the Tab 2 figure renderer**

```python
# add to app.py, after _render_similarity_bars

def _render_verification_figure(
    rows: list[dict],   # each: {label, img_a, img_b, sim, verdict, color}
) -> plt.Figure:
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 3.2))
    fig.patch.set_facecolor("#0f0f1a")
    if n == 1:
        axes = [axes]

    for i, (row, axrow) in enumerate(zip(rows, axes)):
        ax_a, ax_bar, ax_b = axrow
        color = row["color"]

        for ax, img, ttl in [(ax_a, row["img_a"], "Reference A"),
                              (ax_b, row["img_b"], "Test B")]:
            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#1a1a2e")
            ax.set_title(ttl, color="#94a3b8", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(2)

        # bar
        ax_bar.set_facecolor("#0f0f1a")
        ax_bar.barh([0], [row["sim"]], color=color, height=0.4, alpha=0.8)
        ax_bar.axvline(0.50, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.7)
        ax_bar.set_xlim(0, 1.05); ax_bar.set_yticks([])
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        ax_bar.tick_params(colors="#475569")
        ax_bar.set_title(row["label"], color=color, fontsize=10, fontweight="bold")
        ax_bar.text(row["sim"] + 0.01, 0, f'{row["sim"]:.2f}  {row["verdict"]}',
                    va="center", color=color, fontsize=12, fontweight="bold")

    fig.tight_layout(pad=1.5)
    return fig
```

- [ ] **Step 2: Add the Tab 2 event handler**

```python
# add to app.py

def on_verify(photo_a, photo_b):
    img_a = _img_array(photo_a)
    img_b = _img_array(photo_b)
    if img_a is None or img_b is None:
        return gr.update(), "Upload both photos first."

    from attack import GlassesAttacker, Face
    from squeezers import MedianFilterSqueezer
    from insightface.app import FaceAnalysis

    app_if = FaceAnalysis(providers=["CPUExecutionProvider"])
    app_if.prepare(ctx_id=0, det_size=(640, 640))

    embed_a = EMBEDDER.embed(img_a)
    embed_b_clean = EMBEDDER.embed(img_b)
    if embed_a is None or embed_b_clean is None:
        return gr.update(), "Could not detect a face in one of the images."

    # attacked B
    faces_raw = app_if.get(img_b)
    if not faces_raw:
        return gr.update(), "No face found in Photo B for attack."
    face = Face(bbox=faces_raw[0].bbox, kps=faces_raw[0].kps)
    attacked_b = GlassesAttacker(epsilon=0.15).apply(img_b, face)
    embed_b_attacked = EMBEDDER.embed(attacked_b)

    # squeezed attacked B (median kernel=3)
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
            "label":   "Clean A vs Clean B",
            "img_a":   img_a, "img_b": img_b,
            "sim":     sim_clean,
            "verdict": "✅ SAME PERSON" if sim_clean > 0.50 else "❌ DIFFERENT",
            "color":   "#10b981" if sim_clean > 0.50 else "#ef4444",
        },
        {
            "label":   "Clean A vs Attacked B",
            "img_a":   img_a, "img_b": attacked_b,
            "sim":     sim_attacked,
            "verdict": "✅ SAME PERSON" if sim_attacked > 0.50 else "❌ IDENTITY LOST",
            "color":   "#10b981" if sim_attacked > 0.50 else "#ef4444",
        },
        {
            "label":   "Clean A vs Squeezed-Attacked B",
            "img_a":   img_a, "img_b": squeezed_b,
            "sim":     sim_squeezed,
            "verdict": "✅ IDENTITY RECOVERED" if sim_squeezed > 0.50 else "❌ NOT RECOVERED",
            "color":   "#10b981" if sim_squeezed > 0.50 else "#ef4444",
        },
    ]
    fig = _render_verification_figure(rows)
    return fig, ""
```

- [ ] **Step 3: Add Tab 2 layout inside build_app(), inside the gr.Tabs() block**

```python
            # ── TAB 2 ────────────────────────────────────────────────────────
            with gr.Tab("🔍 Verify Identity"):
                gr.HTML("""
                <div style="color:#94a3b8;font-size:12px;padding:8px 0 16px;">
                  Upload two photos of the <strong style="color:#a78bfa;">same person</strong>.
                  See how the adversarial glasses break recognition — and how squeezing restores it.
                </div>
                """)
                with gr.Row():
                    verify_photo_a = gr.Image(label="Photo A — Reference", type="numpy", height=200)
                    verify_photo_b = gr.Image(label="Photo B — Test (same person)", type="numpy", height=200)
                verify_btn = gr.Button("🔍 Run Verification", elem_classes=["btn-defend"])
                verify_error = gr.HTML()
                verify_plot  = gr.Plot(label="Verification Results", show_label=False)

                verify_btn.click(
                    fn=on_verify,
                    inputs=[verify_photo_a, verify_photo_b],
                    outputs=[verify_plot, verify_error],
                )
```

- [ ] **Step 4: Run and test Tab 2 end-to-end**

```bash
python app.py
```
- Switch to the "🔍 Verify Identity" tab
- Upload two photos of the same person (can use `dataset/alice_1.jpg` and `dataset/alice_2.jpg`)
- Click "Run Verification"
- Confirm: 3-row figure appears — row 1 green ✅, row 2 red ❌, row 3 green ✅

- [ ] **Step 5: Run full test suite one last time**

```bash
pytest -v
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: Tab 2 identity verification — clean / attacked / squeezed comparison"
```

---

## Task 12 — Hugging Face Spaces Deployment

**Files:**
- Verify: `README.md` has correct HF Spaces YAML header
- Verify: `requirements.txt` is complete

- [ ] **Step 1: Create the GitHub repository**

```bash
gh repo create adversarial-demo --public --source=. --push
```
Or via GitHub UI: new repo named `adversarial-demo`, then:
```bash
git remote add origin https://github.com/<your-username>/adversarial-demo.git
git push -u origin main
```

- [ ] **Step 2: Create the Hugging Face Space**

Go to https://huggingface.co/new-space:
- Space name: `feature-squeezing-demo`
- SDK: Gradio
- Visibility: Public
- Click "Create Space"

Then link to the GitHub repo under Space Settings → Repository → Connect a GitHub repo, or push directly:

```bash
git remote add space https://huggingface.co/spaces/<your-hf-username>/feature-squeezing-demo
git push space main
```

- [ ] **Step 3: Monitor the build log on HF Spaces**

Open the Space URL → click "Logs" tab. Wait for:
```
Running on public URL: https://<your-hf-username>-feature-squeezing-demo.hf.space
```
First boot takes ~3–4 minutes while insightface downloads model weights.

- [ ] **Step 4: Smoke test the live Space**

Open the public URL. Run through this checklist:
- [ ] Page loads with dark theme
- [ ] Identity dropdown shows alice / bob / carol
- [ ] Select identity → target photo appears
- [ ] Click "⚔ Launch Attack" → pipeline shows original + glasses
- [ ] Click "🛡 Squeeze + Detect" → all 4 panels, bars, verdict banner appear
- [ ] Change bit depth slider → re-run detect → bars update
- [ ] Tab 2: upload 2 photos → verify → 3-row result appears
- [ ] Register a new face (upload 2 photos + name) → appears in dropdown

- [ ] **Step 5: Final commit with Space URL in README**

```bash
# Edit README.md — add after the YAML header:
# 🌐 Live demo: https://<your-hf-username>-feature-squeezing-demo.hf.space
git add README.md
git commit -m "docs: add live Space URL to README"
git push origin main && git push space main
```

---

## Self-Review Checklist

**Spec coverage:**
| Spec requirement | Task |
|---|---|
| Bold & Modern CSS theme | Task 9 |
| Identity dropdown + avatar strip | Task 10 |
| Register face (2 photos, session only) | Task 10 |
| Launch Attack button | Task 10 |
| Bit Depth slider (1–8) | Task 10 |
| Median Kernel slider (3/5/7) | Task 10 |
| Pipeline strip (4 panels + sims) | Task 10 |
| Verdict banner 🚨 / ✅ | Task 10 |
| Same-person verification tab | Task 11 |
| ArcFace via insightface | Task 5 |
| BitDepthSqueezer | Task 2 |
| MedianFilterSqueezer | Task 3 |
| GlassesAttacker (simulated noise) | Task 4 |
| SqueezeDetector (shift > 0.50) | Task 6 |
| IdentityDatabase (3 pre-loaded + 1 live) | Tasks 7 + 8 |
| HF Spaces deployment | Task 12 |

All spec requirements are covered. No gaps found.

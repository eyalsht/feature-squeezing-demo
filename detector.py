from __future__ import annotations
import numpy as np


class FaceDetector:
    """Wraps insightface FaceAnalysis. Lazy-loads model weights on first detect() call."""

    def __init__(self) -> None:
        self._app = None  # lazy

    def _ensure_loaded(self) -> None:
        if self._app is None:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, img: np.ndarray) -> list:
        """Return list of insightface Face objects detected in img (RGB uint8)."""
        self._ensure_loaded()
        return self._app.get(img)


class ArcFaceEmbedder:
    """Extracts L2-normalised 512-dim ArcFace embeddings via a FaceDetector."""

    def __init__(self, detector: FaceDetector) -> None:
        self._detector = detector

    def embed(self, img: np.ndarray) -> np.ndarray | None:
        """Return 512-dim float32 embedding for the first detected face, or None."""
        faces = self._detector.detect(img)
        if not faces:
            return None
        return faces[0].normed_embedding.astype(np.float32)

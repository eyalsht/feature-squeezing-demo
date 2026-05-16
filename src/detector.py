from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from squeezers import BaseSqueezer
from insightface.app import FaceAnalysis


class FaceDetector:
    """Wraps insightface FaceAnalysis for face detection and landmark extraction."""

    def __init__(self) -> None:
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


@dataclass(frozen=True)
class DetectionResult:
    is_adversarial: bool
    max_shift: float
    sims: dict[str, float]
    squeezed_imgs: dict[str, np.ndarray]


class SqueezeDetector:
    """Detects adversarial inputs via Xu et al. (NDSS 2018) self-shift criterion.

    For each squeezer s_i, compute d_i = 1 - cos(embed(x), embed(s_i(x))).
    Flag adversarial when max_i d_i exceeds threshold.
    """

    def __init__(
        self,
        embedder: ArcFaceEmbedder,
        squeezers: list[BaseSqueezer],
        threshold: float = 0.50,
    ) -> None:
        if not squeezers:
            raise ValueError("At least one squeezer required")
        self._embedder = embedder
        self._squeezers = squeezers
        self._threshold = threshold

    def detect(
        self,
        target_embed: np.ndarray,
        original_img: np.ndarray,
        attacked_img: np.ndarray,
    ) -> DetectionResult:
        _orig = self._embedder.embed(original_img)
        embed_original = target_embed if _orig is None else _orig
        _att = self._embedder.embed(attacked_img)
        embed_attacked = target_embed if _att is None else _att

        squeezed_imgs = {sq.name: sq.squeeze(attacked_img) for sq in self._squeezers}
        squeezed_embeds: dict[str, np.ndarray] = {}
        for name, img in squeezed_imgs.items():
            e = self._embedder.embed(img)
            squeezed_embeds[name] = embed_attacked if e is None else e

        sims: dict[str, float] = {
            "original": float(np.dot(target_embed, embed_original)),
            "attacked": float(np.dot(target_embed, embed_attacked)),
        }
        for name, e in squeezed_embeds.items():
            sims[name] = float(np.dot(target_embed, e))

        shifts = [1.0 - float(np.dot(embed_attacked, e)) for e in squeezed_embeds.values()]
        max_shift = max(shifts)

        return DetectionResult(
            is_adversarial=max_shift > self._threshold,
            max_shift=max_shift,
            sims=sims,
            squeezed_imgs=squeezed_imgs,
        )

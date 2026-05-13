from __future__ import annotations
from dataclasses import dataclass
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


from squeezers import BaseSqueezer


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of a SqueezeDetector.detect() call. Immutable."""

    is_adversarial: bool
    max_shift: float
    sims: dict[str, float]
    squeezed_imgs: dict[str, np.ndarray]

    def __repr__(self) -> str:
        verdict = "ADVERSARIAL" if self.is_adversarial else "CLEAN"
        sim_str = ", ".join(f"{k}={v:.3f}" for k, v in self.sims.items())
        return f"DetectionResult({verdict}, shift={self.max_shift:.3f}, [{sim_str}])"


class SqueezeDetector:
    """Detects adversarial inputs via ArcFace embedding shift after feature squeezing.

    Algorithm (Xu et al., NDSS 2018):
        1. Embed the original and attacked images.
        2. Apply each squeezer to the attacked image and embed the result.
        3. Compute cosine similarity of every embedding against `target_embed`.
        4. The per-squeezer shift is |sim_squeezed - sim_attacked|.
        5. Flag the input as adversarial if max(shifts) > threshold.
    """

    _SQUEEZER_KEYS: tuple[str, str] = ("bit", "median")

    def __init__(
        self,
        embedder: ArcFaceEmbedder,
        squeezers: list[BaseSqueezer],
        threshold: float = 0.50,
    ) -> None:
        if len(squeezers) != 2:
            raise ValueError(
                f"Exactly 2 squeezers required (bit, median), got {len(squeezers)}"
            )
        self._embedder = embedder
        self._squeezers = squeezers
        self._threshold = threshold

    def _embed_or_fallback(
        self, img: np.ndarray, fallback: np.ndarray
    ) -> np.ndarray:
        """Embed `img`, returning `fallback` if no face is detected."""
        embed = self._embedder.embed(img)
        return embed if embed is not None else fallback

    def detect(
        self,
        target_embed: np.ndarray,
        original_img: np.ndarray,
        attacked_img: np.ndarray,
    ) -> DetectionResult:
        embed_original = self._embed_or_fallback(original_img, target_embed)
        embed_attacked = self._embed_or_fallback(attacked_img, target_embed)

        squeezed_imgs: dict[str, np.ndarray] = {
            key: sq.squeeze(attacked_img)
            for key, sq in zip(self._SQUEEZER_KEYS, self._squeezers)
        }
        squeezed_embeds: dict[str, np.ndarray] = {
            key: self._embed_or_fallback(img, target_embed)
            for key, img in squeezed_imgs.items()
        }

        sims: dict[str, float] = {
            "original": float(np.dot(target_embed, embed_original)),
            "attacked": float(np.dot(target_embed, embed_attacked)),
            "bit": float(np.dot(target_embed, squeezed_embeds["bit"])),
            "median": float(np.dot(target_embed, squeezed_embeds["median"])),
        }

        sim_attacked = sims["attacked"]
        max_shift = max(abs(sims[k] - sim_attacked) for k in self._SQUEEZER_KEYS)

        return DetectionResult(
            is_adversarial=max_shift > self._threshold,
            max_shift=max_shift,
            sims=sims,
            squeezed_imgs=squeezed_imgs,
        )

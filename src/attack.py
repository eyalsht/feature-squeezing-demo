from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Face:
    bbox: np.ndarray   # shape (4,): [x1, y1, x2, y2]
    kps: np.ndarray    # shape (5, 2): left_eye, right_eye, nose, mouth_l, mouth_r


def make_glasses_mask(img_shape: tuple[int, ...], face: Face) -> np.ndarray:
    """Return a float32 mask (H, W, 1) covering the glasses region of *face*.

    The mask is 1.0 inside the glasses bounding box and 0.0 elsewhere.
    """
    left_eye, right_eye = face.kps[0], face.kps[1]
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    cy = (left_eye[1] + right_eye[1]) / 2

    x1 = max(0, int(left_eye[0] - eye_dist * 0.35))
    x2 = min(img_shape[1], int(right_eye[0] + eye_dist * 0.35))
    y1 = max(0, int(cy - eye_dist * 0.32))
    y2 = min(img_shape[0], int(cy + eye_dist * 0.38))

    mask = np.zeros(img_shape[:2], dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask[:, :, np.newaxis]


class GlassesAttacker:
    def __init__(self, epsilon: float = 0.15) -> None:
        if not 0.0 < epsilon <= 1.0:
            raise ValueError(f"epsilon must be in (0, 1], got {epsilon}")
        self.epsilon = epsilon

    def apply(self, img: np.ndarray, face: Face) -> np.ndarray:
        mask = make_glasses_mask(img.shape, face)
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
        """Deprecated private alias kept for backward-compatibility with tests."""
        return make_glasses_mask(img_shape, face)

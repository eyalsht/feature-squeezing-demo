import numpy as np
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def blank_face_img() -> np.ndarray:
    """160x160 RGB image with a mid-grey face-like gradient."""
    img = np.full((160, 160, 3), 128, dtype=np.uint8)
    cx, cy, r = 80, 80, 60
    y, x = np.ogrid[:160, :160]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    img[mask] = [200, 170, 150]
    return img


@pytest.fixture
def mock_face():
    """Face with eye keypoints at expected positions for a 160x160 image."""
    from attack import Face
    kps = np.array([
        [55.0, 65.0],
        [105.0, 65.0],
        [80.0, 90.0],
        [60.0, 110.0],
        [100.0, 110.0],
    ], dtype=np.float32)
    bbox = np.array([20.0, 30.0, 140.0, 145.0], dtype=np.float32)
    return Face(bbox=bbox, kps=kps)


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    v = np.random.randn(512).astype(np.float32)
    embedder.embed.return_value = v / np.linalg.norm(v)
    return embedder

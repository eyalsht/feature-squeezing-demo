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

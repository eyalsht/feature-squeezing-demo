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


from detector import SqueezeDetector, ArcFaceEmbedder, DetectionResult
from squeezers import BitDepthSqueezer, MedianFilterSqueezer


def _normed(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_squeeze_detector_flags_adversarial_when_shift_exceeds_threshold(blank_face_img):
    target_embed = _normed(np.ones(512))
    original_embed = _normed(np.ones(512))
    attacked_embed = _normed(-np.ones(512))
    squeezed_embed = _normed(np.ones(512) * 0.9)

    mock_embedder = MagicMock(spec=ArcFaceEmbedder)
    mock_embedder.embed.side_effect = [
        original_embed,
        attacked_embed,
        squeezed_embed,
        squeezed_embed,
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

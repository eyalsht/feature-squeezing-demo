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

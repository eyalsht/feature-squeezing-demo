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
    """Loads pre-built identities from disk and supports live registration."""

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

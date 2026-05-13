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

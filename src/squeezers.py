from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from scipy.ndimage import median_filter


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


class MedianFilterSqueezer(BaseSqueezer):
    def __init__(self, kernel: int = 3) -> None:
        if kernel % 2 == 0:
            raise ValueError(f"kernel must be odd, got {kernel}")
        self.kernel = kernel

    def squeeze(self, img: np.ndarray) -> np.ndarray:
        result = np.stack([
            median_filter(img[:, :, c], size=self.kernel)
            for c in range(img.shape[2])
        ], axis=2)
        return result.astype(np.uint8)

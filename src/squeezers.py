from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from scipy.ndimage import median_filter
import cv2


class BaseSqueezer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Short stable identifier used as a dict key in DetectionResult."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-facing label shown in the UI."""

    @abstractmethod
    def squeeze(self, img: np.ndarray) -> np.ndarray:
        """Apply squeezing transform. Input and output are HxWx3 uint8."""


class BitDepthSqueezer(BaseSqueezer):
    name = "bit"

    def __init__(self, bits: int = 4) -> None:
        if not 1 <= bits <= 8:
            raise ValueError(f"bits must be between 1 and 8, got {bits}")
        self.bits = bits

    @property
    def label(self) -> str:
        return f"Bit Depth · {self.bits}"

    def squeeze(self, img: np.ndarray) -> np.ndarray:
        levels = 2 ** self.bits
        if levels == 256:
            return img.copy()
        step = 256 // levels
        quantised = (img.astype(np.uint16) // step) * step
        return np.clip(quantised, 0, 255).astype(np.uint8)


class MedianFilterSqueezer(BaseSqueezer):
    name = "median"

    def __init__(self, kernel: int = 3) -> None:
        if kernel % 2 == 0:
            raise ValueError(f"kernel must be odd, got {kernel}")
        self.kernel = kernel

    @property
    def label(self) -> str:
        return f"Median · {self.kernel}×{self.kernel}"

    def squeeze(self, img: np.ndarray) -> np.ndarray:
        result = np.stack([
            median_filter(img[:, :, c], size=self.kernel, mode="reflect")
            for c in range(img.shape[2])
        ], axis=2)
        return result.astype(np.uint8)


class NonLocalMeansSqueezer(BaseSqueezer):
    """Colour Non-Local Means denoising — the third squeezer recommended by
    Xu et al. for natural images. Defaults mirror EvadeML-Zoo reference."""

    name = "nlm"

    def __init__(
        self,
        strength: int = 11,
        template_window: int = 3,
        search_window: int = 11,
    ) -> None:
        if template_window % 2 == 0 or search_window % 2 == 0:
            raise ValueError("template_window and search_window must be odd")
        if strength <= 0:
            raise ValueError(f"strength must be positive, got {strength}")
        self.strength = strength
        self.template_window = template_window
        self.search_window = search_window

    @property
    def label(self) -> str:
        return f"NLM · h={self.strength}"

    def squeeze(self, img: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(
            img,
            None,
            float(self.strength),
            float(self.strength),
            self.template_window,
            self.search_window,
        )

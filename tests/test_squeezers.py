import numpy as np
import pytest
from squeezers import BitDepthSqueezer


def test_bit_depth_squeezer_reduces_unique_values():
    img = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
    img = np.repeat(img, 3, axis=2)
    squeezer = BitDepthSqueezer(bits=2)
    result = squeezer.squeeze(img)
    assert result.dtype == np.uint8
    unique_vals = np.unique(result)
    assert len(unique_vals) <= 4   # 2^2 = 4 levels


def test_bit_depth_squeezer_preserves_shape(blank_face_img):
    squeezer = BitDepthSqueezer(bits=4)
    result = squeezer.squeeze(blank_face_img)
    assert result.shape == blank_face_img.shape


def test_bit_depth_squeezer_1_bit_gives_two_values():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = BitDepthSqueezer(bits=1).squeeze(img)
    assert set(np.unique(result)).issubset({0, 128})


def test_bit_depth_squeezer_8_bits_is_identity():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = BitDepthSqueezer(bits=8).squeeze(img)
    np.testing.assert_array_equal(result, img)

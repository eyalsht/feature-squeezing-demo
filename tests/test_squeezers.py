import numpy as np
import pytest
from squeezers import BitDepthSqueezer, MedianFilterSqueezer


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


def test_median_squeezer_removes_salt_and_pepper(blank_face_img):
    noisy = blank_face_img.copy()
    rng = np.random.default_rng(42)
    coords = rng.integers(0, 160, size=(200, 2))
    noisy[coords[:, 0], coords[:, 1]] = 255
    squeezer = MedianFilterSqueezer(kernel=3)
    result = squeezer.squeeze(noisy)
    diff_noisy = np.mean(np.abs(noisy.astype(int) - blank_face_img.astype(int)))
    diff_result = np.mean(np.abs(result.astype(int) - blank_face_img.astype(int)))
    assert diff_result < diff_noisy


def test_median_squeezer_preserves_shape(blank_face_img):
    result = MedianFilterSqueezer(kernel=5).squeeze(blank_face_img)
    assert result.shape == blank_face_img.shape
    assert result.dtype == np.uint8


def test_median_squeezer_rejects_even_kernel():
    with pytest.raises(ValueError, match="odd"):
        MedianFilterSqueezer(kernel=4)

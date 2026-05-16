import numpy as np
import pytest
from attack import Face, GlassesAttacker, make_glasses_mask


def test_apply_changes_eye_region(blank_face_img, mock_face):
    attacker = GlassesAttacker(epsilon=0.15)
    result = attacker.apply(blank_face_img, mock_face)
    y1, y2 = 50, 80
    eye_region_orig = blank_face_img[y1:y2, 40:120]
    eye_region_result = result[y1:y2, 40:120]
    assert not np.array_equal(eye_region_orig, eye_region_result)


def test_apply_preserves_shape_and_dtype(blank_face_img, mock_face):
    result = GlassesAttacker(epsilon=0.10).apply(blank_face_img, mock_face)
    assert result.shape == blank_face_img.shape
    assert result.dtype == np.uint8


def test_apply_stays_in_valid_pixel_range(blank_face_img, mock_face):
    result = GlassesAttacker(epsilon=0.99).apply(blank_face_img, mock_face)
    assert result.min() >= 0
    assert result.max() <= 255


def test_make_glasses_mask_covers_eye_region(mock_face):
    attacker = GlassesAttacker()
    mask = attacker._make_glasses_mask((160, 160, 3), mock_face)
    assert mask.shape == (160, 160, 1)
    eye_y = int(mock_face.kps[:2, 1].mean())
    assert mask[eye_y, 80, 0] == 1.0
    assert mask[155, 80, 0] == 0.0


def test_module_level_make_glasses_mask_importable_and_correct_shape(mock_face):
    """make_glasses_mask must be importable as a module-level public function."""
    mask = make_glasses_mask((160, 160, 3), mock_face)
    assert mask.shape == (160, 160, 1), "mask should have shape (H, W, 1)"
    eye_y = int(mock_face.kps[:2, 1].mean())
    assert mask[eye_y, 80, 0] == 1.0, "eye-centre pixel should be masked"
    assert mask[155, 80, 0] == 0.0, "chin pixel should not be masked"

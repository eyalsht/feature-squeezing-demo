"""TDD tests for src/attack_lab.py — ImpersonationAttacker.

Run order: RED (these fail before implementation) → GREEN → REFACTOR.
Uses mocked ArcFaceEmbedder to keep tests CPU-cheap and fast.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 1: basic import
# ---------------------------------------------------------------------------

def test_attack_lab_imports():
    """AttackResult and ImpersonationAttacker must be importable."""
    from attack_lab import AttackResult, ImpersonationAttacker  # noqa: F401
    assert True


# ---------------------------------------------------------------------------
# Test 2: parameter validation
# ---------------------------------------------------------------------------

def test_attacker_init_validates_params():
    """Negative lr or non-positive steps must raise ValueError."""
    from attack_lab import ImpersonationAttacker
    with pytest.raises(ValueError):
        ImpersonationAttacker(lr=-0.01)
    with pytest.raises(ValueError):
        ImpersonationAttacker(steps=0)
    with pytest.raises(ValueError):
        ImpersonationAttacker(steps=-5)


# ---------------------------------------------------------------------------
# Test 3: pixels outside mask are unchanged
# ---------------------------------------------------------------------------

def test_pixels_outside_mask_unchanged(mocker):
    """Pixels outside the glasses mask must not change after the attack."""
    from attack_lab import ImpersonationAttacker

    H, W = 64, 64
    rng = np.random.default_rng(42)
    attacker_img = rng.integers(80, 180, (H, W, 3), dtype=np.uint8)
    target_img  = rng.integers(80, 180, (H, W, 3), dtype=np.uint8)

    # Small mask in the top-left quadrant only
    mask = np.zeros((H, W, 1), dtype=np.float32)
    mask[10:30, 10:40] = 1.0

    # Mock the ArcFaceEmbedder used internally for cross-model evaluation
    mock_embedder = mocker.MagicMock()
    vec = rng.standard_normal(512).astype(np.float32)
    mock_embedder.embed.return_value = vec / np.linalg.norm(vec)

    attacker = ImpersonationAttacker(steps=3, lr=0.02)
    result = attacker.attack(attacker_img, target_img, mask,
                             arc_embedder=mock_embedder)

    outside = mask[:, :, 0] == 0.0
    np.testing.assert_array_equal(
        result.attacked_img[outside],
        attacker_img[outside],
        err_msg="Pixels outside the mask must not change",
    )


# ---------------------------------------------------------------------------
# Test 4: result shape
# ---------------------------------------------------------------------------

def test_attack_returns_result_shape(mocker):
    """attack() must return an AttackResult with correctly-shaped arrays."""
    from attack_lab import ImpersonationAttacker, AttackResult

    H, W = 64, 64
    rng = np.random.default_rng(7)
    attacker_img = rng.integers(50, 200, (H, W, 3), dtype=np.uint8)
    target_img   = rng.integers(50, 200, (H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W, 1), dtype=np.float32)
    mask[15:35, 15:50] = 1.0

    mock_embedder = mocker.MagicMock()
    vec = rng.standard_normal(512).astype(np.float32)
    mock_embedder.embed.return_value = vec / np.linalg.norm(vec)

    attacker = ImpersonationAttacker(steps=3, lr=0.02)
    result = attacker.attack(attacker_img, target_img, mask,
                             arc_embedder=mock_embedder)

    assert isinstance(result, AttackResult)
    assert result.attacked_img.shape == (H, W, 3)
    assert result.attacked_img.dtype == np.uint8
    assert result.glasses_pattern.shape[2] == 3  # HxWx3, cropped to mask bbox
    assert isinstance(result.sim_before, float)
    assert isinstance(result.sim_after, float)
    assert isinstance(result.sim_after_torch, float)
    assert isinstance(result.converged, bool)
    assert isinstance(result.iterations_run, int)
    assert result.iterations_run == 3


# ---------------------------------------------------------------------------
# Test 5: converged flag consistent
# ---------------------------------------------------------------------------

def test_converged_flag_consistent(mocker):
    """converged must be True iff sim_after > sim_before."""
    from attack_lab import ImpersonationAttacker

    H, W = 64, 64
    rng = np.random.default_rng(99)
    attacker_img = rng.integers(50, 200, (H, W, 3), dtype=np.uint8)
    target_img   = rng.integers(50, 200, (H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W, 1), dtype=np.float32)
    mask[10:30, 10:40] = 1.0

    mock_embedder = mocker.MagicMock()
    vec = rng.standard_normal(512).astype(np.float32)
    mock_embedder.embed.return_value = vec / np.linalg.norm(vec)

    attacker = ImpersonationAttacker(steps=3, lr=0.02)
    result = attacker.attack(attacker_img, target_img, mask,
                             arc_embedder=mock_embedder)

    # converged definition: sim_after > sim_before
    expected = result.sim_after > result.sim_before
    assert result.converged == expected, (
        f"converged={result.converged} but sim_before={result.sim_before:.4f},"
        f" sim_after={result.sim_after:.4f}"
    )

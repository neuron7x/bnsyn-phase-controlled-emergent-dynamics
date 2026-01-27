"""Smoke tests for sleep replay utilities.

Parameters
----------
None

Returns
-------
None

Notes
-----
Tests pattern selection and noise addition for memory replay.

References
----------
docs/sleep_stack.md
"""

from __future__ import annotations

import numpy as np
import pytest

from bnsyn.rng import seed_all
from bnsyn.sleep.replay import add_replay_noise, weighted_pattern_selection


def test_weighted_pattern_selection() -> None:
    """Test importance-weighted pattern selection."""
    pack = seed_all(42)
    rng = pack.np_rng

    patterns = [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ]
    importance = np.array([0.1, 0.8, 0.1], dtype=np.float64)

    # Select multiple times and check that high importance pattern is selected more
    selections = []
    for _ in range(100):
        selected = weighted_pattern_selection(patterns, importance, rng)
        selections.append(np.argmax(selected))

    # Pattern 1 (importance 0.8) should be selected most often
    assert selections.count(1) > selections.count(0)
    assert selections.count(1) > selections.count(2)


def test_weighted_pattern_selection_uniform_weights() -> None:
    """Test selection with zero importance falls back to uniform."""
    pack = seed_all(42)
    rng = pack.np_rng

    patterns = [
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    ]
    importance = np.array([0.0, 0.0], dtype=np.float64)

    # Should work with zero importance (uniform selection)
    selected = weighted_pattern_selection(patterns, importance, rng)
    assert selected.shape == (2,)


def test_weighted_pattern_selection_validation() -> None:
    """Test input validation for pattern selection."""
    pack = seed_all(42)
    rng = pack.np_rng

    patterns = [np.array([1.0, 0.0], dtype=np.float64)]
    importance = np.array([0.5], dtype=np.float64)

    # Empty patterns
    with pytest.raises(ValueError, match="patterns list is empty"):
        weighted_pattern_selection([], importance, rng)

    # Mismatched importance length
    with pytest.raises(ValueError, match="importance length must match"):
        weighted_pattern_selection(patterns, np.array([0.5, 0.3], dtype=np.float64), rng)


def test_add_replay_noise_zero_noise() -> None:
    """Test adding noise with noise_level=0."""
    pack = seed_all(42)
    rng = pack.np_rng

    pattern = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    noisy = add_replay_noise(pattern, noise_level=0.0, noise_scale=10.0, rng=rng)

    # Should return exact copy
    np.testing.assert_array_equal(noisy, pattern)


def test_add_replay_noise_with_noise() -> None:
    """Test adding noise with non-zero noise_level."""
    pack = seed_all(42)
    rng = pack.np_rng

    pattern = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    noisy = add_replay_noise(pattern, noise_level=0.5, noise_scale=1.0, rng=rng)

    # Should be different from original
    assert not np.array_equal(noisy, pattern)
    # Should be roughly similar (not too far)
    assert np.abs(noisy - pattern).mean() < 2.0


def test_add_replay_noise_determinism() -> None:
    """Test that noise addition is deterministic with same seed."""
    pattern = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    # First run
    pack1 = seed_all(123)
    noisy1 = add_replay_noise(pattern, noise_level=0.5, noise_scale=1.0, rng=pack1.np_rng)

    # Second run
    pack2 = seed_all(123)
    noisy2 = add_replay_noise(pattern, noise_level=0.5, noise_scale=1.0, rng=pack2.np_rng)

    # Should be identical
    np.testing.assert_array_equal(noisy1, noisy2)


def test_add_replay_noise_validation() -> None:
    """Test input validation for noise addition."""
    pack = seed_all(42)
    rng = pack.np_rng
    pattern = np.array([1.0, 2.0], dtype=np.float64)

    # Invalid noise_level
    with pytest.raises(ValueError, match="noise_level must be in"):
        add_replay_noise(pattern, noise_level=1.5, noise_scale=1.0, rng=rng)

    with pytest.raises(ValueError, match="noise_level must be in"):
        add_replay_noise(pattern, noise_level=-0.1, noise_scale=1.0, rng=rng)


def test_weighted_pattern_selection_determinism() -> None:
    """Test that pattern selection is deterministic with same seed."""
    patterns = [
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    ]
    importance = np.array([0.3, 0.7], dtype=np.float64)

    # First run
    pack1 = seed_all(456)
    selections1 = []
    for _ in range(10):
        selected = weighted_pattern_selection(patterns, importance, pack1.np_rng)
        selections1.append(np.argmax(selected))

    # Second run
    pack2 = seed_all(456)
    selections2 = []
    for _ in range(10):
        selected = weighted_pattern_selection(patterns, importance, pack2.np_rng)
        selections2.append(np.argmax(selected))

    # Should be identical
    assert selections1 == selections2

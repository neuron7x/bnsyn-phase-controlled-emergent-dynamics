"""Deterministic RNG utilities for BN-Syn.

Implements SPEC P2-9 determinism protocol with explicit seeding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RNGPack:
    """Bundle RNG artifacts produced by seeding.

    Args:
        seed: Seed value used to initialize RNGs.
        np_rng: NumPy Generator instance.
    """

    seed: int
    np_rng: np.random.Generator


def seed_all(seed: int) -> RNGPack:
    """Seed all RNGs used by this project.

    Args:
        seed: Integer seed in [0, 2**32 - 1].

    Returns:
        RNGPack containing the NumPy Generator and seed.

    Raises:
        TypeError: If seed is not an int.
        ValueError: If seed is outside the allowed range.

    Notes:
        Sets PYTHONHASHSEED and returns an explicit NumPy Generator.

    References:
        - docs/SPEC.md#P2-9
        - docs/REPRODUCIBILITY.md
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be int")
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("seed must be in [0, 2**32-1]")

    os.environ["PYTHONHASHSEED"] = str(seed)
    np_rng = np.random.default_rng(seed)
    return RNGPack(seed=seed, np_rng=np_rng)


def split(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Deterministically split a generator into child generators.

    Args:
        rng: Source NumPy Generator.
        n: Number of child generators.

    Returns:
        List of NumPy Generators seeded deterministically.

    Raises:
        ValueError: If n is non-positive.

    Notes:
        Uses integer draws from the parent generator to seed children.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    seeds = rng.integers(0, 2**32 - 1, size=n, dtype=np.uint32)
    return [np.random.default_rng(int(s)) for s in seeds]

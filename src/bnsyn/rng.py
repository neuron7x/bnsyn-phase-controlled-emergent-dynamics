"""Random number generation utilities for deterministic BN-Syn runs."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RNGPack:
    """Container for seeded RNG state."""

    seed: int
    np_rng: np.random.Generator


def seed_all(seed: int) -> RNGPack:
    """Seed all RNGs used by this project.

    Parameters
    ----------
    seed
        Seed in the 32-bit unsigned integer range.

    Returns
    -------
    RNGPack
        Seeded RNG container with a NumPy generator.

    Raises
    ------
    TypeError
        If ``seed`` is not an integer.
    ValueError
        If ``seed`` is outside ``[0, 2**32 - 1]``.

    Notes
    -----
    NumPy is the primary RNG. Python's ``random`` is also seeded to avoid
    accidental nondeterminism in downstream code.
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be int")
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("seed must be in [0, 2**32-1]")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np_rng = np.random.default_rng(seed)
    return RNGPack(seed=seed, np_rng=np_rng)


def split(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Deterministically split a generator into ``n`` generators.

    Parameters
    ----------
    rng
        Source generator to split.
    n
        Number of independent generators to create.

    Returns
    -------
    list[numpy.random.Generator]
        List of independent generators derived from ``rng``.

    Raises
    ------
    ValueError
        If ``n`` is not positive.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    seeds = rng.integers(0, 2**32 - 1, size=n, dtype=np.uint32)
    return [np.random.default_rng(int(s)) for s in seeds]

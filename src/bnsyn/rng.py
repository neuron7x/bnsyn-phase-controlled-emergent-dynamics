"""Deterministic RNG utilities for BN-Syn.

Parameters
----------
None

Returns
-------
None

Determinism
-----------
All RNG state is derived from explicit seeds and returned as NumPy generators.

SPEC
----
SPEC.md §P2-9

Claims
------
CLM-0023
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RNGPack:
    """Bundle RNG artifacts produced by seeding.

    Parameters
    ----------
    seed : int
        Seed value used to initialize RNGs.
    np_rng : numpy.random.Generator
        NumPy Generator instance.

    Returns
    -------
    RNGPack
        Container with the seed and generator.

    Determinism
    -----------
    Deterministic under fixed seed.

    SPEC
    ----
    SPEC.md §P2-9

    Claims
    ------
    CLM-0023
    """

    seed: int
    np_rng: np.random.Generator


def seed_all(seed: int) -> RNGPack:
    """Seed all RNGs used by this project.

    Parameters
    ----------
    seed : int
        Integer seed in [0, 2**32 - 1].

    Returns
    -------
    RNGPack
        Container holding the seed and NumPy Generator.

    Determinism
    -----------
    Deterministic under fixed integer seed; sets ``PYTHONHASHSEED`` and returns an explicit
    NumPy ``Generator`` instance.

    SPEC
    ----
    SPEC.md §P2-9

    Claims
    ------
    CLM-0023
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

    Parameters
    ----------
    rng : numpy.random.Generator
        Source NumPy generator.
    n : int
        Number of child generators to create.

    Returns
    -------
    list[numpy.random.Generator]
        Child generators seeded deterministically from the parent.

    Determinism
    -----------
    Deterministic under fixed parent generator state.

    SPEC
    ----
    SPEC.md §P2-9

    Claims
    ------
    CLM-0023
    """
    if n <= 0:
        raise ValueError("n must be positive")
    seeds = rng.integers(0, 2**32 - 1, size=n, dtype=np.uint32)
    return [np.random.default_rng(int(s)) for s in seeds]

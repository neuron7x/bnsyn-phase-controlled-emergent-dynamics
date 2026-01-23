from __future__ import annotations

import os
import random
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RNGPack:
    seed: int
    np_rng: np.random.Generator


def seed_all(seed: int) -> RNGPack:
    """Seed all RNGs used by this project.

    Note: This package is NumPy-first. We seed Python's `random` to avoid
    accidental nondeterminism if user code uses it.
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
    """Deterministically split a generator into `n` generators."""
    if n <= 0:
        raise ValueError("n must be positive")
    seeds = rng.integers(0, 2**32 - 1, size=n, dtype=np.uint32)
    return [np.random.default_rng(int(s)) for s in seeds]

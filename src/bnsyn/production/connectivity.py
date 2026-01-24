"""Deterministic connectivity builders for BN-Syn experiments.

Parameters
----------
None

Returns
-------
None

Determinism
-----------
Deterministic under fixed RNG state.

SPEC
----
SPEC.md §P2-11

Claims
------
None
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class ConnectivityConfig:
    """Connectivity configuration for deterministic builders.

    Parameters
    ----------
    n_pre : int
        Number of presynaptic units.
    n_post : int
        Number of postsynaptic units.
    p_connect : float
        Connection probability.
    allow_self : bool
        Allow self-connections when n_pre == n_post.

    Returns
    -------
    ConnectivityConfig
        Configuration instance.

    Determinism
    -----------
    Deterministic data container.

    SPEC
    ----
    SPEC.md §P2-11

    Claims
    ------
    None
    """
    n_pre: int
    n_post: int
    p_connect: float
    allow_self: bool = False


def build_connectivity(
    cfg: ConnectivityConfig,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a boolean adjacency matrix with Bernoulli(p_connect).

    Parameters
    ----------
    cfg : ConnectivityConfig
        Connectivity configuration.
    rng : numpy.random.Generator
        NumPy Generator for deterministic sampling.

    Returns
    -------
    numpy.ndarray
        Boolean adjacency matrix.

    Determinism
    -----------
    Deterministic under fixed RNG state.

    SPEC
    ----
    SPEC.md §P2-11, §P2-9

    Claims
    ------
    CLM-0023
    """
    if cfg.n_pre <= 0 or cfg.n_post <= 0:
        raise ValueError("n_pre and n_post must be > 0")
    if not (0.0 <= cfg.p_connect <= 1.0):
        raise ValueError("p_connect must be in [0, 1]")
    adj = rng.random((cfg.n_post, cfg.n_pre)) < cfg.p_connect
    if not cfg.allow_self and cfg.n_pre == cfg.n_post:
        np.fill_diagonal(adj, False)
    return adj

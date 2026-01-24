"""Reciprocity gating utilities for VCG-style support signals.

Parameters
----------
None

Returns
-------
None

Determinism
-----------
Deterministic under fixed inputs.

SPEC
----
SPEC.md §GOV-1

Claims
------
CLM-0015, CLM-0016, CLM-0017, CLM-0018
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VCGParams:
    """Parameter bundle for VCG-inspired support gating.

    Parameters
    ----------
    theta_c : float
        Contribution threshold for support updates.
    alpha_down : float
        Support decrement for insufficient contribution.
    alpha_up : float
        Support increment for adequate contribution.
    epsilon : float
        Minimum allocation multiplier.

    Returns
    -------
    VCGParams
        Parameter container instance.

    Determinism
    -----------
    Deterministic data container.

    SPEC
    ----
    SPEC.md §GOV-1

    Claims
    ------
    CLM-0015, CLM-0016, CLM-0017, CLM-0018
    """
    theta_c: float
    alpha_down: float
    alpha_up: float
    epsilon: float

    def __post_init__(self) -> None:
        if self.theta_c < 0.0:
            raise ValueError("theta_c must be non-negative")
        if self.alpha_down < 0.0 or self.alpha_up < 0.0:
            raise ValueError("alpha_down and alpha_up must be non-negative")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")


def update_support_level(contribution: float, support: float, params: VCGParams) -> float:
    """Update support level deterministically based on contribution threshold.

    Parameters
    ----------
    contribution : float
        Contribution value.
    support : float
        Current support level in [0, 1].
    params : VCGParams
        VCG parameter set.

    Returns
    -------
    float
        Updated support level.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §GOV-1

    Claims
    ------
    CLM-0015, CLM-0016
    """
    if not 0.0 <= support <= 1.0:
        raise ValueError("support must be in [0, 1]")
    if contribution < params.theta_c:
        return float(max(0.0, support - params.alpha_down))
    return float(min(1.0, support + params.alpha_up))


def allocation_multiplier(support: float, params: VCGParams) -> float:
    """Compute allocation multiplier from support level.

    Parameters
    ----------
    support : float
        Support level in [0, 1].
    params : VCGParams
        VCG parameter set.

    Returns
    -------
    float
        Allocation multiplier.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §GOV-1

    Claims
    ------
    CLM-0017, CLM-0018
    """
    if not 0.0 <= support <= 1.0:
        raise ValueError("support must be in [0, 1]")
    return float(params.epsilon + (1.0 - params.epsilon) * support)


def update_support_vector(
    contributions: np.ndarray, support: np.ndarray, params: VCGParams
) -> np.ndarray:
    """Vectorized support update for multiple agents.

    Parameters
    ----------
    contributions : numpy.ndarray
        Contribution values.
    support : numpy.ndarray
        Current support levels in [0, 1].
    params : VCGParams
        VCG parameter set.

    Returns
    -------
    numpy.ndarray
        Updated support levels.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §GOV-1

    Claims
    ------
    CLM-0017, CLM-0018
    """
    if contributions.shape != support.shape:
        raise ValueError("contributions and support must have the same shape")
    updated = np.where(
        contributions < params.theta_c,
        np.maximum(0.0, support - params.alpha_down),
        np.minimum(1.0, support + params.alpha_up),
    )
    return updated.astype(float)

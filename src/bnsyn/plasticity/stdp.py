"""Spike-timing dependent plasticity kernel utilities.

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
SPEC.md §P0-3

Claims
------
CLM-0005
"""

from __future__ import annotations

import numpy as np

from bnsyn.config import PlasticityParams


def stdp_kernel(delta_t_ms: float, p: PlasticityParams) -> float:
    """Izhikevich-style exponential STDP kernel.

    Parameters
    ----------
    delta_t_ms : float
        Spike timing difference (t_post - t_pre) in milliseconds.
    p : PlasticityParams
        Plasticity parameter set.

    Returns
    -------
    float
        STDP kernel value.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P0-3

    Claims
    ------
    CLM-0005
    """
    if delta_t_ms > 0:
        return float(p.A_plus * np.exp(-delta_t_ms / p.tau_plus_ms))
    if delta_t_ms < 0:
        return float(-p.A_minus * np.exp(delta_t_ms / p.tau_minus_ms))
    return 0.0

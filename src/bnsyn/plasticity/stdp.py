from __future__ import annotations

import numpy as np

from bnsyn.config import PlasticityParams


def stdp_kernel(delta_t_ms: float, p: PlasticityParams) -> float:
    """Izhikevich-style exponential STDP kernel.

    delta_t = t_post - t_pre (ms). Positive => LTP.
    """
    if delta_t_ms > 0:
        return float(p.A_plus * np.exp(-delta_t_ms / p.tau_plus_ms))
    if delta_t_ms < 0:
        return float(-p.A_minus * np.exp(delta_t_ms / p.tau_minus_ms))
    return 0.0

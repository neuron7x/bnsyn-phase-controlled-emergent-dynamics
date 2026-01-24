"""Criticality estimation and homeostatic gain control.

Parameters
----------
None

Returns
-------
None

Determinism
-----------
Deterministic under fixed inputs and fixed parameters.

SPEC
----
SPEC.md §P0-4

Claims
------
CLM-0006, CLM-0007
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bnsyn.config import CriticalityParams


@dataclass
class BranchingEstimator:
    """Estimate branching ratio sigma with exponential smoothing.

    Parameters
    ----------
    eps : float
        Guard epsilon to avoid divide-by-zero.
    ema_alpha : float
        EMA smoothing factor.

    Returns
    -------
    BranchingEstimator
        Estimator instance.

    Determinism
    -----------
    Deterministic given fixed inputs.

    SPEC
    ----
    SPEC.md §P0-4

    Claims
    ------
    CLM-0007
    """

    eps: float = 1e-9
    ema_alpha: float = 0.05
    _sigma_ema: float = 1.0

    def update(self, A_t: float, A_t1: float) -> float:
        """Update sigma estimate based on consecutive activity counts.

        Parameters
        ----------
        A_t : float
            Activity at time t (spike count).
        A_t1 : float
            Activity at time t+1 (spike count).

        Returns
        -------
        float
            Smoothed sigma estimate.

        Determinism
        -----------
        Deterministic given fixed inputs.

        SPEC
        ----
        SPEC.md §P0-4

        Claims
        ------
        CLM-0007
        """
        A_t = float(A_t)
        A_t1 = float(A_t1)
        sigma = A_t1 / max(A_t, self.eps)
        self._sigma_ema = (1.0 - self.ema_alpha) * self._sigma_ema + self.ema_alpha * sigma
        return float(self._sigma_ema)


@dataclass
class SigmaController:
    """Homeostatic gain controller for sigma regulation.

    Parameters
    ----------
    params : CriticalityParams
        Criticality parameter set.
    gain : float
        Initial gain value.

    Returns
    -------
    SigmaController
        Controller instance.

    Determinism
    -----------
    Deterministic given fixed inputs.

    SPEC
    ----
    SPEC.md §P0-4

    Claims
    ------
    CLM-0006
    """

    params: CriticalityParams
    gain: float = 1.0

    def step(self, sigma: float) -> float:
        """Update the gain given the current sigma estimate.

        Parameters
        ----------
        sigma : float
            Current sigma estimate.

        Returns
        -------
        float
            Updated gain value.

        Determinism
        -----------
        Deterministic given fixed inputs.

        SPEC
        ----
        SPEC.md §P0-4

        Claims
        ------
        CLM-0006
        """
        p = self.params
        self.gain = float(self.gain - p.eta_sigma * (float(sigma) - p.sigma_target))
        self.gain = float(np.clip(self.gain, p.gain_min, p.gain_max))
        return self.gain

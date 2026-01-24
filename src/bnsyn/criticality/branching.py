"""Criticality estimators and gain controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bnsyn.config import CriticalityParams


@dataclass
class BranchingEstimator:
    """Finite-size branching estimator ``σ_t = A(t+1)/A(t)``."""

    eps: float = 1e-9
    ema_alpha: float = 0.05
    _sigma_ema: float = 1.0

    def update(self, A_t: float, A_t1: float) -> float:
        """Update the branching ratio estimate.

        Parameters
        ----------
        A_t
            Activity at time ``t``.
        A_t1
            Activity at time ``t+1``.

        Returns
        -------
        float
            Updated exponential moving average of sigma.
        """
        A_t = float(A_t)
        A_t1 = float(A_t1)
        sigma = A_t1 / max(A_t, self.eps)
        self._sigma_ema = (1.0 - self.ema_alpha) * self._sigma_ema + self.ema_alpha * sigma
        return float(self._sigma_ema)


@dataclass
class SigmaController:
    """Homeostatic gain controller for criticality regulation."""

    params: CriticalityParams
    gain: float = 1.0

    def step(self, sigma: float) -> float:
        """Update gain based on the current sigma estimate.

        Parameters
        ----------
        sigma
            Current branching ratio estimate.

        Returns
        -------
        float
            Updated gain value.
        """
        p = self.params
        self.gain = float(self.gain - p.eta_sigma * (float(sigma) - p.sigma_target))
        self.gain = float(np.clip(self.gain, p.gain_min, p.gain_max))
        return self.gain

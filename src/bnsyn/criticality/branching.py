from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bnsyn.config import CriticalityParams


@dataclass
class BranchingEstimator:
    """Finite-size branching estimator σ_t = A(t+1)/A(t) with epsilon guard.

    For subsampling-corrected estimators (e.g. MR), see docs/SPEC.md; this is a
    deterministic baseline used in CI.
    """

    eps: float = 1e-9
    ema_alpha: float = 0.05
    _sigma_ema: float = 1.0

    def update(self, A_t: float, A_t1: float) -> float:
        A_t = float(A_t)
        A_t1 = float(A_t1)
        sigma = A_t1 / max(A_t, self.eps)
        self._sigma_ema = (1.0 - self.ema_alpha) * self._sigma_ema + self.ema_alpha * sigma
        return float(self._sigma_ema)


@dataclass
class SigmaController:
    """Homeostatic gain controller: Γ <- Γ - η(σ-σ*) clipped to bounds."""

    params: CriticalityParams
    gain: float = 1.0

    def step(self, sigma: float) -> float:
        p = self.params
        self.gain = float(self.gain - p.eta_sigma * (float(sigma) - p.sigma_target))
        self.gain = float(np.clip(self.gain, p.gain_min, p.gain_max))
        return self.gain

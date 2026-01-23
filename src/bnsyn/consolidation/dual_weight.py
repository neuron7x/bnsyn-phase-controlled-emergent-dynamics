from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bnsyn.config import DualWeightParams


@dataclass
class DualWeights:
    """Dual-weight synapse model: w_total = w_fast + w_cons.

    - Fast weights decay to baseline w0 on tau_f.
    - Tag set when |w_fast - w0| > theta_tag.
    - Protein is a global scalar synthesised when enough tags active.
    - Consolidated weights follow slow tracking when Tag & Protein.
    """

    w_fast: np.ndarray
    w_cons: np.ndarray
    w0: float
    tags: np.ndarray
    protein: float

    @classmethod
    def init(cls, shape: tuple[int, int], w0: float = 0.0) -> "DualWeights":
        return cls(
            w_fast=np.full(shape, w0, dtype=float),
            w_cons=np.full(shape, w0, dtype=float),
            w0=float(w0),
            tags=np.zeros(shape, dtype=bool),
            protein=0.0,
        )

    def step(
        self,
        dt_s: float,
        p: DualWeightParams,
        fast_update: np.ndarray,
    ) -> None:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if fast_update.shape != self.w_fast.shape:
            raise ValueError("fast_update shape mismatch")

        # fast dynamics: update + decay to baseline
        self.w_fast += p.eta_f * fast_update
        self.w_fast += (-(self.w_fast - self.w0) / p.tau_f_s) * dt_s

        # tag setting
        self.tags = np.abs(self.w_fast - self.w0) > p.theta_tag

        # protein synthesis (cooperative): if enough tags are set, protein increases towards 1
        tag_count = float(np.sum(self.tags))
        N_p = 50.0  # default cooperative threshold; documented in SPEC.md
        synth = 1.0 if tag_count >= N_p else 0.0
        self.protein += (synth * (1.0 - self.protein) - self.protein / p.tau_p_s) * dt_s
        self.protein = float(np.clip(self.protein, 0.0, 1.0))

        # consolidation: slow tracking towards w_fast when Tag & Protein
        mask = self.tags.astype(float) * self.protein
        self.w_cons += p.eta_c * (self.w_fast - self.w_cons) * mask * dt_s

    @property
    def w_total(self) -> np.ndarray:
        return self.w_fast + self.w_cons

"""Dual-weight consolidation dynamics for synapses.

Parameters
----------
None

Returns
-------
None

Determinism
-----------
Deterministic under fixed inputs and fixed timestep.

SPEC
----
SPEC.md §P1-6

Claims
------
CLM-0010, CLM-0020
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bnsyn.config import DualWeightParams


@dataclass
class DualWeights:
    """Dual-weight synapse model: w_total = w_fast + w_cons.

    Parameters
    ----------
    w_fast : numpy.ndarray
        Fast weight matrix.
    w_cons : numpy.ndarray
        Consolidated weight matrix.
    w0 : float
        Baseline weight.
    tags : numpy.ndarray
        Tag indicators for consolidation.
    protein : float
        Global protein availability scalar.

    Returns
    -------
    DualWeights
        Dual-weight state container.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P1-6

    Claims
    ------
    CLM-0010, CLM-0020
    """

    w_fast: np.ndarray
    w_cons: np.ndarray
    w0: float
    tags: np.ndarray
    protein: float

    @classmethod
    def init(cls, shape: tuple[int, int], w0: float = 0.0) -> "DualWeights":
        """Initialize a dual-weight container with baseline weights.

        Parameters
        ----------
        shape : tuple[int, int]
            Matrix shape for the weights.
        w0 : float
            Baseline weight value.

        Returns
        -------
        DualWeights
            Initialized dual-weight container.

        Determinism
        -----------
        Deterministic given fixed inputs.

        SPEC
        ----
        SPEC.md §P1-6

        Claims
        ------
        CLM-0010, CLM-0020
        """
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
        """Advance dual-weight dynamics by one timestep.

        Parameters
        ----------
        dt_s : float
            Timestep in seconds.
        p : DualWeightParams
            Consolidation parameter set.
        fast_update : numpy.ndarray
            Fast weight update increment.

        Returns
        -------
        None

        Determinism
        -----------
        Deterministic under fixed inputs.

        SPEC
        ----
        SPEC.md §P1-6

        Claims
        ------
        CLM-0010, CLM-0020
        """
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if fast_update.shape != self.w_fast.shape:
            raise ValueError("fast_update shape mismatch")

        # fast dynamics: update + decay to baseline
        self.w_fast += p.eta_f * fast_update * dt_s
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
        """Return total weights as the sum of fast and consolidated components.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Total weight matrix.

        Determinism
        -----------
        Deterministic under fixed state.

        SPEC
        ----
        SPEC.md §P1-6

        Claims
        ------
        CLM-0010, CLM-0020
        """
        return np.asarray(self.w_fast + self.w_cons)

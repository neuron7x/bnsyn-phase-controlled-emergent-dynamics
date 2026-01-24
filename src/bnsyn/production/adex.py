"""Adaptive Exponential Integrate-and-Fire (AdEx) neuron model (NumPy).

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
SPEC.md §P0-1

Claims
------
None
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True, slots=True)
class AdExParams:
    """Parameter bundle for the production AdEx helper.

    Parameters
    ----------
    C : float
        Membrane capacitance (F).
    gL : float
        Leak conductance (S).
    EL : float
        Leak reversal potential (V).
    VT : float
        Threshold voltage (V).
    DeltaT : float
        Slope factor (V).
    tau_w : float
        Adaptation time constant (s).
    a : float
        Subthreshold adaptation conductance (S).
    b : float
        Spike-triggered adaptation increment (A).
    V_reset : float
        Reset voltage (V).
    V_spike : float
        Spike threshold (V).
    t_ref : float
        Refractory period (s).

    Returns
    -------
    AdExParams
        Parameter container.

    Determinism
    -----------
    Deterministic data container.

    SPEC
    ----
    SPEC.md §P0-1

    Claims
    ------
    None
    """
    # Membrane
    C: float = 90e-12  # Farads
    gL: float = 10e-9  # Siemens
    EL: float = -70e-3  # Volts
    VT: float = -50e-3  # Volts
    DeltaT: float = 2e-3  # Volts

    # Adaptation
    tau_w: float = 100e-3  # Seconds
    a: float = 2e-9  # Siemens
    b: float = 0.0  # Amps

    # Spike/reset
    V_reset: float = -58e-3  # Volts
    V_spike: float = 0.0  # Volts (detection threshold)

    # Refractory
    t_ref: float = 5e-3  # Seconds


@dataclass(slots=True)
class AdExNeuron:
    """Vectorized AdEx neuron state.

    Parameters
    ----------
    params : AdExParams
        AdEx parameter set.
    V : numpy.ndarray
        Membrane potentials (V).
    w : numpy.ndarray
        Adaptation currents (A).
    t_last_spike : numpy.ndarray
        Last spike times (s).

    Returns
    -------
    AdExNeuron
        Neuron state container.

    Determinism
    -----------
    Deterministic under fixed inputs.

    SPEC
    ----
    SPEC.md §P0-1

    Claims
    ------
    None
    """

    params: AdExParams
    V: np.ndarray
    w: np.ndarray
    t_last_spike: np.ndarray

    @classmethod
    def init(
        cls, n: int, params: AdExParams | None = None, *, V0: float | None = None
    ) -> "AdExNeuron":
        """Initialize a vectorized AdEx neuron state.

        Parameters
        ----------
        n : int
            Number of neurons.
        params : AdExParams | None
            Optional parameter set.
        V0 : float | None
            Optional initial voltage (V).

        Returns
        -------
        AdExNeuron
            Initialized neuron state.

        Determinism
        -----------
        Deterministic under fixed inputs.

        SPEC
        ----
        SPEC.md §P0-1

        Claims
        ------
        None
        """
        p = params or AdExParams()
        v0 = p.EL if V0 is None else float(V0)
        V = np.full((n,), v0, dtype=np.float64)
        w = np.zeros((n,), dtype=np.float64)
        t_last_spike = np.full((n,), -np.inf, dtype=np.float64)
        return cls(params=p, V=V, w=w, t_last_spike=t_last_spike)

    def step(self, input_current: np.ndarray, dt: float, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Advance state by one Euler step.

        Parameters
        ----------
        input_current : numpy.ndarray
            Input current (A), shape (n,).
        dt : float
            Timestep (s).
        t : float
            Current simulation time (s), used for refractory tracking.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Spike indicators and updated membrane potentials.

        Determinism
        -----------
        Deterministic under fixed inputs.

        SPEC
        ----
        SPEC.md §P0-1

        Claims
        ------
        None
        """
        p = self.params
        current = np.asarray(input_current, dtype=np.float64)
        if current.shape != self.V.shape:
            raise ValueError(
                f"input_current shape {current.shape} must match V shape {self.V.shape}"
            )

        # Refractory: hold at reset
        in_ref = (t - self.t_last_spike) < p.t_ref
        V_eff = np.where(in_ref, p.V_reset, self.V)

        exp_term = p.gL * p.DeltaT * np.exp((V_eff - p.VT) / p.DeltaT)
        dV = (-(p.gL * (V_eff - p.EL)) + exp_term - self.w + current) / p.C
        dw = (p.a * (V_eff - p.EL) - self.w) / p.tau_w

        V_new = V_eff + dt * dV
        w_new = self.w + dt * dw
        V_new = np.where(in_ref, p.V_reset, V_new)

        spikes = V_new >= p.V_spike
        if np.any(spikes):
            V_new = np.where(spikes, p.V_reset, V_new)
            w_new = np.where(spikes, w_new + p.b, w_new)
            self.t_last_spike = np.where(spikes, t, self.t_last_spike)

        self.V = V_new
        self.w = w_new
        return spikes, self.V

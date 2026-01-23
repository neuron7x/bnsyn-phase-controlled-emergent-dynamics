from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bnsyn.config import SynapseParams
from bnsyn.numerics.integrators import exp_decay_step

Float64Array = NDArray[np.float64]


@dataclass
class ConductanceState:
    g_ampa_nS: Float64Array
    g_nmda_nS: Float64Array
    g_gabaa_nS: Float64Array


def nmda_mg_block(V_mV: Float64Array, mg_mM: float) -> Float64Array:
    """Jahr-Stevens Mg2+ block: B(V) = 1 / (1 + ([Mg]/3.57)*exp(-0.062 V))."""
    return np.asarray(1.0 / (1.0 + (mg_mM / 3.57) * np.exp(-0.062 * V_mV)), dtype=np.float64)


class ConductanceSynapses:
    """Event-driven conductance synapses with fixed delay using a ring buffer.

    This class does not implement connectivity; it applies aggregate incoming spikes.
    Upstream code supplies per-neuron incoming spike counts (or weighted counts).
    """

    def __init__(self, N: int, params: SynapseParams, dt_ms: float) -> None:
        if N <= 0:
            raise ValueError("N must be positive")
        if dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        self.N = N
        self.params = params
        self.dt_ms = dt_ms

        delay_steps = max(1, int(round(params.delay_ms / dt_ms)))
        self._delay_steps = delay_steps
        self._buf = np.zeros((delay_steps, N), dtype=np.float64)
        self._buf_idx = 0

        self.state = ConductanceState(
            g_ampa_nS=np.zeros(N, dtype=np.float64),
            g_nmda_nS=np.zeros(N, dtype=np.float64),
            g_gabaa_nS=np.zeros(N, dtype=np.float64),
        )

    @property
    def delay_steps(self) -> int:
        return self._delay_steps

    def queue_events(self, incoming: Float64Array) -> None:
        """Queue incoming events to be applied after the fixed delay.

        `incoming` is shape (N,) and represents aggregate conductance increments.
        Units are nS increments per timestep (already includes weights).
        """
        if incoming.shape != (self.N,):
            raise ValueError(f"incoming must have shape ({self.N},)")
        self._buf[self._buf_idx, :] = np.asarray(incoming, dtype=np.float64)

    def step(self) -> Float64Array:
        """Advance synaptic state by one dt and return I_syn in pA for each neuron."""
        # apply delayed events (written delay_steps ago)
        apply = self._buf[self._buf_idx, :].copy()
        self._buf[self._buf_idx, :] = 0.0
        self._buf_idx = (self._buf_idx + 1) % self._delay_steps

        # split apply into receptor types (simple convention: 60% AMPA, 30% NMDA, 10% GABA_A)
        # This is an architecture choice; for explicit networks provide three vectors instead.
        self.state.g_ampa_nS += 0.6 * apply
        self.state.g_nmda_nS += 0.3 * apply
        self.state.g_gabaa_nS += 0.1 * apply

        p = self.params
        dt = self.dt_ms
        self.state.g_ampa_nS = exp_decay_step(self.state.g_ampa_nS, dt, p.tau_AMPA_ms)
        self.state.g_nmda_nS = exp_decay_step(self.state.g_nmda_nS, dt, p.tau_NMDA_ms)
        self.state.g_gabaa_nS = exp_decay_step(self.state.g_gabaa_nS, dt, p.tau_GABAA_ms)

        # I_syn = g*(V - E) terms are computed outside (needs V). Here return conductances.
        # We return a tuple-like stacked array for downstream. For convenience, return (3,N).
        return np.stack(
            [self.state.g_ampa_nS, self.state.g_nmda_nS, self.state.g_gabaa_nS], axis=0
        )

    @staticmethod
    def current_pA(
        V_mV: Float64Array,
        g_ampa_nS: Float64Array,
        g_nmda_nS: Float64Array,
        g_gabaa_nS: Float64Array,
        params: SynapseParams,
    ) -> Float64Array:
        B = nmda_mg_block(V_mV, params.mg_mM)
        current = (
            g_ampa_nS * (V_mV - params.E_AMPA_mV)
            + g_nmda_nS * B * (V_mV - params.E_NMDA_mV)
            + g_gabaa_nS * (V_mV - params.E_GABAA_mV)
        )
        # nS*mV => pA
        return np.asarray(current, dtype=np.float64)

"""Reference network simulator for BN-Syn.

Implements SPEC P2-11 reference network dynamics for deterministic tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os
import numpy as np

from bnsyn.config import AdExParams, CriticalityParams, SynapseParams
from bnsyn.connectivity import SparseConnectivity
from bnsyn.criticality.branching import BranchingEstimator, SigmaController
from bnsyn.neuron.adex import AdExState, adex_step, adex_step_adaptive
from bnsyn.numerics.integrators import exp_decay_step
from bnsyn.synapse.conductance import nmda_mg_block
from bnsyn.validation import NetworkValidationConfig, validate_connectivity_matrix

torch: Any | None
try:
    import torch as torch_module  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional GPU support
    torch = None
else:
    torch = torch_module


@dataclass(frozen=True)
class NetworkParams:
    """Network configuration parameters.

    Args:
        N: Number of neurons.
        frac_inhib: Fraction of inhibitory neurons (0, 1).
        p_conn: Connection probability for random connectivity.
        w_exc_nS: Excitatory synaptic weight in nS.
        w_inh_nS: Inhibitory synaptic weight in nS.
        ext_rate_hz: External Poisson drive rate per neuron (Hz).
        ext_w_nS: External synaptic weight in nS.
        V_min_mV: Minimum membrane voltage bound (mV).
        V_max_mV: Maximum membrane voltage bound (mV).
    """

    N: int = 200
    frac_inhib: float = 0.2
    p_conn: float = 0.05
    w_exc_nS: float = 0.5
    w_inh_nS: float = 1.0
    ext_rate_hz: float = 2.0  # Poisson external drive per neuron
    ext_w_nS: float = 0.3

    # Simulation bounds
    V_min_mV: float = -100.0
    V_max_mV: float = 50.0


class Network:
    """Small reference network (dense enough for tests, not optimized).

    Args:
        nparams: Network configuration parameters.
        adex: AdEx neuron parameters.
        syn: Synapse parameters.
        crit: Criticality control parameters.
        dt_ms: Timestep in milliseconds.
        rng: NumPy RNG for deterministic sampling.

    Raises:
        ValueError: If parameters are invalid.

    Notes:
        Implements SPEC P2-11 and integrates SPEC P0-1, P0-2, P0-4 components.

    References:
        - docs/SPEC.md#P2-11
        - docs/SSOT.md
    """

    def __init__(
        self,
        nparams: NetworkParams,
        adex: AdExParams,
        syn: SynapseParams,
        crit: CriticalityParams,
        dt_ms: float,
        rng: np.random.Generator,
    ):
        if nparams.N <= 0:
            raise ValueError("N must be positive")
        if not (0.0 < nparams.frac_inhib < 1.0):
            raise ValueError("frac_inhib must be in (0,1)")
        if dt_ms <= 0:
            raise ValueError("dt_ms must be positive")

        self.np = nparams
        self.adex = adex
        self.syn = syn
        self.dt_ms = dt_ms
        self.rng = rng

        N = nparams.N
        nI = int(round(N * nparams.frac_inhib))
        nE = N - nI
        self.nE, self.nI = nE, nI
        self.is_inhib = np.zeros(N, dtype=bool)
        self.is_inhib[nE:] = True

        # adjacency masks
        mask = rng.random((N, N)) < nparams.p_conn
        np.fill_diagonal(mask, False)
        # excitatory weights (E->*)
        W_exc = np.asarray((mask[:nE, :].astype(np.float64) * nparams.w_exc_nS).T)
        # inhibitory weights (I->*)
        W_inh = np.asarray((mask[nE:, :].astype(np.float64) * nparams.w_inh_nS).T)

        validate_connectivity_matrix(W_exc, shape=(N, nE), name="W_exc")
        validate_connectivity_matrix(W_inh, shape=(N, nI), name="W_inh")

        self.W_exc = SparseConnectivity(W_exc)
        self.W_inh = SparseConnectivity(W_inh)

        # neuron state
        V0 = np.asarray(rng.normal(loc=adex.EL_mV, scale=5.0, size=N), dtype=np.float64)
        w0 = np.zeros(N, dtype=np.float64)
        self.state = AdExState(V_mV=V0, w_pA=w0, spiked=np.zeros(N, dtype=bool))

        # conductances
        self.g_ampa = np.zeros(N, dtype=np.float64)
        self.g_nmda = np.zeros(N, dtype=np.float64)
        self.g_gabaa = np.zeros(N, dtype=np.float64)

        # criticality tracking
        self.branch = BranchingEstimator()
        self.sigma_ctl = SigmaController(params=crit, gain=1.0)
        self.gain = 1.0
        self._A_prev = 1.0
        self._use_torch = False
        self._torch_device = None
        self._W_exc_t = None
        self._W_inh_t = None

        if torch is not None and os.environ.get("BNSYN_USE_TORCH") == "1":
            device = os.environ.get("BNSYN_DEVICE", "cpu")
            self._torch_device = torch.device(device)
            self._W_exc_t = torch.as_tensor(self.W_exc.to_dense(), device=self._torch_device)
            self._W_inh_t = torch.as_tensor(self.W_inh.to_dense(), device=self._torch_device)
            self._use_torch = True

    def step(self) -> dict[str, float]:
        """Advance the network by one timestep.

        Returns:
            Dictionary of metrics including sigma, gain, and spike rate.

        Raises:
            RuntimeError: If voltage bounds are violated (numerical instability).

        Notes:
            Criticality gain is updated each step using sigma tracking.
        """
        N = self.np.N
        dt = self.dt_ms

        # external Poisson spikes (rate per neuron)
        lam = self.np.ext_rate_hz * (dt / 1000.0)
        ext_spikes = self.rng.random(N) < lam
        incoming_ext = ext_spikes.astype(float) * self.np.ext_w_nS

        # recurrent contributions from last step spikes
        spikes = self.state.spiked
        spikes_E = spikes[: self.nE].astype(float)
        spikes_I = spikes[self.nE :].astype(float)

        if self._use_torch:
            assert torch is not None
            spikes_E_t = torch.as_tensor(spikes_E, dtype=torch.float64, device=self._torch_device)
            spikes_I_t = torch.as_tensor(spikes_I, dtype=torch.float64, device=self._torch_device)
            incoming_exc = torch.matmul(self._W_exc_t, spikes_E_t).cpu().numpy()
            incoming_inh = torch.matmul(self._W_inh_t, spikes_I_t).cpu().numpy()
        else:
            incoming_exc = self.W_exc.apply(np.asarray(spikes_E, dtype=np.float64))
            incoming_inh = self.W_inh.apply(np.asarray(spikes_I, dtype=np.float64))

        # apply increments (split E into AMPA/NMDA)
        self.g_ampa += 0.7 * incoming_exc + 0.7 * incoming_ext
        self.g_nmda += 0.3 * incoming_exc + 0.3 * incoming_ext
        self.g_gabaa += incoming_inh

        # decay (exponential, dt-invariant)
        self.g_ampa = exp_decay_step(self.g_ampa, dt, self.syn.tau_AMPA_ms)
        self.g_nmda = exp_decay_step(self.g_nmda, dt, self.syn.tau_NMDA_ms)
        self.g_gabaa = exp_decay_step(self.g_gabaa, dt, self.syn.tau_GABAA_ms)

        # compute synaptic current (pA), then scale excitability by gain (criticality controller)
        V = self.state.V_mV
        B = nmda_mg_block(V, self.syn.mg_mM)
        I_syn = (
            self.g_ampa * (V - self.syn.E_AMPA_mV)
            + self.g_nmda * B * (V - self.syn.E_NMDA_mV)
            + self.g_gabaa * (V - self.syn.E_GABAA_mV)
        )

        # gain: multiplies external current (proxy for global excitability)
        I_ext = np.zeros(N, dtype=np.float64)
        I_ext += 50.0 * (self.gain - 1.0)  # pA offset

        self.state = adex_step(self.state, self.adex, dt, I_syn_pA=I_syn, I_ext_pA=I_ext)

        # safety bounds
        if (
            float(np.min(self.state.V_mV)) < self.np.V_min_mV
            or float(np.max(self.state.V_mV)) > self.np.V_max_mV
        ):
            raise RuntimeError("Voltage bounds violation (numerical instability)")

        # criticality estimation from population activity
        A_t = float(np.sum(spikes))
        A_t1 = float(np.sum(self.state.spiked))
        sigma = self.branch.update(A_t=max(A_t, 1.0), A_t1=max(A_t1, 1.0))
        self.gain = self.sigma_ctl.step(sigma)

        self._A_prev = A_t1

        return {
            "A_t": A_t,
            "A_t1": A_t1,
            "sigma": float(sigma),
            "gain": float(self.gain),
            "spike_rate_hz": float(A_t1 / N) / (dt / 1000.0),
        }

    def step_adaptive(self, *, atol: float = 1e-8, rtol: float = 1e-6) -> dict[str, float]:
        """Advance the network by one timestep using adaptive AdEx integration.

        Args:
            atol: Absolute tolerance for adaptive AdEx integration.
            rtol: Relative tolerance for adaptive AdEx integration.

        Returns:
            Dictionary of metrics including sigma, gain, and spike rate.

        Raises:
            RuntimeError: If voltage bounds are violated (numerical instability).
        """
        N = self.np.N
        dt = self.dt_ms

        lam = self.np.ext_rate_hz * (dt / 1000.0)
        ext_spikes = self.rng.random(N) < lam
        incoming_ext = ext_spikes.astype(float) * self.np.ext_w_nS

        spikes = self.state.spiked
        spikes_E = spikes[: self.nE].astype(float)
        spikes_I = spikes[self.nE :].astype(float)

        if self._use_torch:
            assert torch is not None
            spikes_E_t = torch.as_tensor(spikes_E, dtype=torch.float64, device=self._torch_device)
            spikes_I_t = torch.as_tensor(spikes_I, dtype=torch.float64, device=self._torch_device)
            incoming_exc = torch.matmul(self._W_exc_t, spikes_E_t).cpu().numpy()
            incoming_inh = torch.matmul(self._W_inh_t, spikes_I_t).cpu().numpy()
        else:
            incoming_exc = self.W_exc.apply(np.asarray(spikes_E, dtype=np.float64))
            incoming_inh = self.W_inh.apply(np.asarray(spikes_I, dtype=np.float64))

        self.g_ampa += 0.7 * incoming_exc + 0.7 * incoming_ext
        self.g_nmda += 0.3 * incoming_exc + 0.3 * incoming_ext
        self.g_gabaa += incoming_inh

        self.g_ampa = exp_decay_step(self.g_ampa, dt, self.syn.tau_AMPA_ms)
        self.g_nmda = exp_decay_step(self.g_nmda, dt, self.syn.tau_NMDA_ms)
        self.g_gabaa = exp_decay_step(self.g_gabaa, dt, self.syn.tau_GABAA_ms)

        V = self.state.V_mV
        B = nmda_mg_block(V, self.syn.mg_mM)
        I_syn = (
            self.g_ampa * (V - self.syn.E_AMPA_mV)
            + self.g_nmda * B * (V - self.syn.E_NMDA_mV)
            + self.g_gabaa * (V - self.syn.E_GABAA_mV)
        )

        I_ext = np.zeros(N, dtype=np.float64)
        I_ext += 50.0 * (self.gain - 1.0)

        self.state = adex_step_adaptive(
            self.state,
            self.adex,
            dt,
            I_syn_pA=I_syn,
            I_ext_pA=I_ext,
            atol=atol,
            rtol=rtol,
        )

        if (
            float(np.min(self.state.V_mV)) < self.np.V_min_mV
            or float(np.max(self.state.V_mV)) > self.np.V_max_mV
        ):
            raise RuntimeError("Voltage bounds violation (numerical instability)")

        A_t = float(np.sum(spikes))
        A_t1 = float(np.sum(self.state.spiked))
        sigma = self.branch.update(A_t=max(A_t, 1.0), A_t1=max(A_t1, 1.0))
        self.gain = self.sigma_ctl.step(sigma)

        self._A_prev = A_t1

        return {
            "A_t": A_t,
            "A_t1": A_t1,
            "sigma": float(sigma),
            "gain": float(self.gain),
            "spike_rate_hz": float(A_t1 / N) / (dt / 1000.0),
        }


def run_simulation(
    steps: int,
    dt_ms: float,
    seed: int,
    N: int = 200,
) -> dict[str, float]:
    """Run a deterministic simulation and return summary metrics.

    Args:
        steps: Number of simulation steps.
        dt_ms: Timestep in milliseconds.
        seed: RNG seed.
        N: Number of neurons.

    Returns:
        Summary metrics with mean and standard deviation for sigma and firing rate.

    References:
        - docs/SPEC.md#P2-11
        - docs/REPRODUCIBILITY.md
    """
    from bnsyn.rng import seed_all

    _ = NetworkValidationConfig(N=N, dt_ms=dt_ms)
    pack = seed_all(seed)
    rng = pack.np_rng
    nparams = NetworkParams(N=N)
    net = Network(nparams, AdExParams(), SynapseParams(), CriticalityParams(), dt_ms=dt_ms, rng=rng)

    sigmas: list[float] = []
    rates: list[float] = []
    for _ in range(steps):
        m = net.step()
        sigmas.append(m["sigma"])
        rates.append(m["spike_rate_hz"])

    return {
        "sigma_mean": float(np.mean(sigmas)),
        "rate_mean_hz": float(np.mean(rates)),
        "sigma_std": float(np.std(sigmas)),
        "rate_std": float(np.std(rates)),
    }

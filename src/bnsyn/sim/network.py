from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bnsyn.config import AdExParams, CriticalityParams, SynapseParams
from bnsyn.criticality.branching import BranchingEstimator, SigmaController
from bnsyn.neuron.adex import AdExState, adex_step
from bnsyn.numerics.integrators import exp_decay_step
from bnsyn.synapse.conductance import nmda_mg_block


@dataclass(frozen=True)
class NetworkParams:
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
    """Small reference network (dense enough for tests, not optimized)."""

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
        self.W_exc = (mask[:nE, :].astype(float) * nparams.w_exc_nS).T  # shape (N, nE)
        # inhibitory weights (I->*)
        self.W_inh = (mask[nE:, :].astype(float) * nparams.w_inh_nS).T  # shape (N, nI)

        # neuron state
        V0 = rng.normal(loc=adex.EL_mV, scale=5.0, size=N)
        w0 = np.zeros(N, dtype=float)
        self.state = AdExState(V_mV=V0, w_pA=w0, spiked=np.zeros(N, dtype=bool))

        # conductances
        self.g_ampa = np.zeros(N, dtype=float)
        self.g_nmda = np.zeros(N, dtype=float)
        self.g_gabaa = np.zeros(N, dtype=float)

        # criticality tracking
        self.branch = BranchingEstimator()
        self.sigma_ctl = SigmaController(params=crit, gain=1.0)
        self.gain = 1.0
        self._A_prev = 1.0

    def step(self) -> dict[str, float]:
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

        incoming_exc = self.W_exc @ spikes_E  # nS increments
        incoming_inh = self.W_inh @ spikes_I  # nS increments

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
        I_ext = np.zeros(N, dtype=float)
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


def run_simulation(
    steps: int,
    dt_ms: float,
    seed: int,
    N: int = 200,
) -> dict[str, float]:
    from bnsyn.rng import seed_all

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

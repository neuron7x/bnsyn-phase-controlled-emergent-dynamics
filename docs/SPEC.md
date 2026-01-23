# BN-Syn Thermostated Bio-AI System — Formal Specification (v0.2.0)

This document is the **authoritative** spec for the reference implementation in `src/bnsyn/`.
It is structured as **12 components** with: (a) equations, (b) calibration knobs, (c) failure envelopes,
(d) acceptance checks implemented in `tests/`.

> Scope: The repo implements a **minimal but complete** reference system suitable for deterministic CI.
> Scaling to large-N / GPU is explicitly out-of-scope for v0.2.0 but interfaces are stable.

---

## Component Index (12)

| ID | Component | Implementation | Primary Tests |
|---:|---|---|---|
| P0-1 | AdEx neuron dynamics | `bnsyn/neuron/adex.py` | `test_adex_smoke.py` |
| P0-2 | Conductance synapses (AMPA/NMDA/GABA_A) + NMDA Mg block | `bnsyn/synapse/conductance.py` | `test_synapse_smoke.py` |
| P0-3 | Three-factor learning (eligibility × neuromodulator) | `bnsyn/plasticity/three_factor.py` | `test_plasticity_smoke.py` |
| P0-4 | Criticality tracking σ and gain homeostasis | `bnsyn/criticality/branching.py` | `test_criticality_smoke.py` |
| P1-5 | Temperature schedule + gating | `bnsyn/temperature/schedule.py` | `test_temperature_smoke.py` |
| P1-6 | Dual-weight consolidation (fast/slow + tags + protein) | `bnsyn/consolidation/dual_weight.py` | `test_consolidation_smoke.py` |
| P1-7 | Energy regularization objective terms | `bnsyn/energy/regularization.py` | `test_energy_smoke.py` |
| P2-8 | Numerical methods (Euler/RK2/exp decay) | `bnsyn/numerics/integrators.py` | `test_dt_invariance.py` |
| P2-9 | Determinism protocol (seed + explicit RNG) | `bnsyn/rng.py` | `test_determinism.py` |
| P2-10 | Calibration utilities (f–I fit) | `bnsyn/calibration/fit.py` | `test_calibration_smoke.py` |
| P2-11 | Reference network simulator (small-N) | `bnsyn/sim/network.py` | `test_network_smoke.py` |
| P2-12 | Bench harness contract (CLI + metrics) | `bnsyn/cli.py` | `test_cli_smoke.py` |

---

## P0-1: AdEx neuron (Brette & Gerstner, 2005)

Membrane equation:
\[
C\frac{dV}{dt}=-g_L(V-E_L)+g_L\Delta_T\exp\left(\frac{V-V_T}{\Delta_T}\right)-w-I_{syn}+I_{ext}
\]

Adaptation:
\[
\tau_w\frac{dw}{dt}=a(V-E_L)-w
\]

Reset at spike:
\[
V\to V_r,\quad w\to w+b
\]

**Implementation note:** numerical overflow is avoided by clamping the exponential argument to 20.

Failure envelope:
- Euler unstable when `dt_ms` approaches membrane time-constant scale.
- Bound guard: `V ∈ [-100, +50] mV` enforced by `Network.step()`.

---

## P0-2: Conductance synapses + NMDA Mg²⁺ block (Jahr–Stevens, 1990)

Synaptic current:
\[
I_{syn}=g_{AMPA}(V-E_{AMPA})+g_{NMDA}B(V)(V-E_{NMDA})+g_{GABA_A}(V-E_{GABA_A})
\]

Mg block:
\[
B(V)=\frac{1}{1+\frac{[Mg^{2+}]_o}{3.57}\exp(-0.062V)}
\]

Decay uses exponential update:
\[
g(t+\Delta t)=g(t)\exp(-\Delta t/\tau)
\]
which is unconditionally stable and supports Δt-invariance tests.

---

## P0-3: Three-factor learning (Frémaux & Gerstner, 2016)

Eligibility trace:
\[
\frac{de_{ij}}{dt}=-\frac{e_{ij}}{\tau_e}+\text{coincidence}_{ij}
\]

Weight update:
\[
\Delta w_{ij}=\eta\,e_{ij}\,M
\]

**v0.2.0 simplification:** coincidence is approximated by an outer product of pre/post spike indicators at each timestep.
Full spike-time STDP (tracking last spike times and kernel evaluation) is a planned enhancement.

Hard bounds:
\[
w\in[w_{min}, w_{max}]
\]

---

## P0-4: Criticality σ and gain control

Branching estimate:
\[
\sigma_t = \frac{A(t+1)}{A(t)}
\]
with EMA smoothing for deterministic CI.

Gain homeostasis:
\[
\Gamma \leftarrow \Gamma - \eta_\sigma(\sigma-\sigma^*)
\]
clipped to \([\Gamma_{min},\Gamma_{max}]\).

---

## P1-5: Temperature schedule + gating

Geometric cooling:
\[
T_{k+1}=\alpha T_k
\]

Plasticity gate (sigmoid):
\[
G(T)=\frac{1}{1+\exp(-(T-T_c)/\tau)}
\]

Used to couple exploration→consolidation in downstream systems.

---

## P1-6: Dual-weight consolidation (STC-inspired)

\[
w_{total}=w^f+w^c
\]

Fast weights decay to baseline:
\[
\frac{dw^f}{dt} = \eta_f\,u - \frac{w^f-w_0}{\tau_f}
\]

Tag: \(\mathbb{1}(|w^f-w_0|>\theta_{tag})\). Consolidation tracks fast weights only when Tag & Protein.

---

## P2-8..12: Validation, determinism, Δt-invariance

See `docs/REPRODUCIBILITY.md` and `tests/test_dt_invariance.py`.

---

## Primary sources

- Brette & Gerstner (2005) AdEx. *J Neurophysiol* 94:3637–3642.
- Jahr & Stevens (1990) NMDA Mg block. *J Neurosci* 10:3178–3182.
- Frémaux & Gerstner (2016) Three-factor learning. *Front Neural Circuits* 9:85.

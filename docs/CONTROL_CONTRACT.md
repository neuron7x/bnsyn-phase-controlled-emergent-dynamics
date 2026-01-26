# CONTROL CONTRACT — BN-Syn System Boundaries

## Overview

This document defines the control-theoretic boundaries of the BN-Syn thermostated bio-AI system. It specifies inputs, observables, stability criteria, and failure modes from a control systems engineering perspective.

---

## System Inputs (Control Variables)

### Temperature Schedule `T(t)`
- **Type**: Time-varying scalar, continuous
- **Range**: `[0.1, 2.0]` (dimensionless, relative to baseline)
- **Purpose**: Modulates synaptic plasticity rate via temperature-gated consolidation
- **Implementation**: `bnsyn.temperature.TemperatureSchedule`
- **Control Law**: Piecewise constant, linear ramps, or custom functions

### Sigma Controller Parameters
- **Type**: Feedback controller configuration
- **Variables**:
  - `sigma_target`: Target normalized standard deviation of membrane potentials (criticality setpoint)
  - `k_p`, `k_i`, `k_d`: PID gains for synaptic scaling
  - `update_interval`: Controller sample period
- **Purpose**: Maintain network near criticality via homeostatic synaptic scaling
- **Implementation**: `bnsyn.control.SigmaController`

### External Drive `I_ext(t, i)`
- **Type**: Time- and neuron-indexed current injection
- **Range**: `[-∞, ∞]` pA (typically `[-50, 50]`)
- **Purpose**: Task-specific input patterns, sensory stimulation
- **Default**: Constant background drive + optional noise

### Network Topology
- **Type**: Sparse connectivity matrix (E-E, E-I, I-E, I-I blocks)
- **Variables**:
  - `p_conn`: Connection probability per block
  - `g_syn`: Synaptic conductance strength scale
  - `N_E`, `N_I`: Excitatory/inhibitory population sizes
- **Invariant**: Fixed at initialization (no structural plasticity)
- **Implementation**: `bnsyn.connectivity.sparse.SparseConnectivity`

---

## System Observables (Measurements)

### Instantaneous State
- `V[i]`: Membrane potential of neuron `i` (mV)
- `w[i]`: Adaptation current of neuron `i` (pA, AdEx model)
- `s[j]`: Synaptic conductance state (nS)
- `W[j]`: Plastic synaptic weight (fast component)
- `W_cons[j]`: Consolidated weight (slow, protein-mediated)

### Aggregate Dynamics
- `sigma`: Standard deviation of membrane potentials (criticality proxy)
- `spike_rate`: Network-averaged firing rate (Hz)
- `g_mean`: Mean excitatory synaptic strength
- `sync_index`: Spike time synchrony measure
- `avalanche_size`: Distribution of burst sizes

### Consolidation Metrics
- `protein_level`: Normalized protein synthesis indicator `[0, 1]`
- `w_total_variance`: Variance of total weights `W + W_cons`
- `w_cons_variance`: Variance of consolidated weights alone
- `cons_rate`: Rate of fast→slow weight transfer

### Phase Indicators
- `criticality_phase`: {SUBCRITICAL, CRITICAL, SUPERCRITICAL, UNKNOWN}
- `sleep_stage`: {AWAKE, NREM, REM} (if sleep module active)

---

## Stability Criteria

### 1. Boundedness
**Requirement**: All state variables remain bounded for all `t ∈ [0, T_sim]`

**Constraints**:
- `V[i] ∈ [-100, 50]` mV (physiological range)
- `w[i] ≥ 0` pA (adaptation cannot be negative)
- `W[j] ∈ [0, W_max]` (non-negative weights, finite ceiling)
- `sigma ∈ [0, σ_max]` (standard deviation cannot diverge)

**Violation**: Simulation halt with error (numerical instability)

### 2. Deterministic Replay
**Requirement**: Identical initial conditions + seed → identical trajectory

**Enforcement**:
- All randomness via `RNGPack` with explicit seeding
- No `np.random.*` or `random.*` in application code
- No wall-clock or system-state dependencies

**Validation**: Golden hash test (`tests/test_golden_hash.py`)

### 3. Criticality Maintenance (Optional Control Loop)
**Requirement**: If sigma controller enabled, `sigma` converges to `sigma_target ± δ`

**Tolerance**: `δ = 0.05` (5% of target)
**Timescale**: Convergence within ~500 ms (depends on PID tuning)

**Failure Mode**: Divergence to supercritical (epileptiform) or subcritical (silent network)

### 4. Consolidation Stability
**Requirement**: Under appropriate cooling, `w_total_variance` decreases or stabilizes

**Non-Triviality Check**: `protein_level > 0.5` (consolidation active, not suppressed)

**Validation**: Multi-seed statistical tests (validation suite)

---

## Failure Modes

### Numerical Instability
**Symptoms**:
- `NaN` or `Inf` in state variables
- `V` exceeds physiological bounds

**Causes**:
- Timestep `dt` too large (CFL violation)
- Synaptic strengths too strong (runaway excitation)
- Integration error accumulation

**Mitigation**:
- Adaptive timestep (future work)
- Synaptic strength bounds enforcement
- Sigma controller as safety fallback

### Non-Determinism
**Symptoms**:
- Different results with same seed
- Hash mismatch in golden hash test

**Causes**:
- Unseeded randomness
- Floating-point non-associativity (parallel sum order)
- System clock or file system dependencies

**Mitigation**:
- Strict RNG discipline (no global state)
- Deterministic aggregation (sorted reductions)
- CI golden hash enforcement

### Import Failure (Missing Optional Deps)
**Symptoms**:
- `ModuleNotFoundError` when importing visualization
- CI failure in minimal environment

**Causes**:
- Direct `import matplotlib` instead of lazy loading
- `__init__.py` eagerly importing optional submodules

**Mitigation**:
- `importlib.import_module()` for optional deps
- Test coverage: `tests/test_viz_optional_import.py`
- CI without `[viz]` extra

### Test Tier Violation
**Symptoms**:
- PR CI slow (>2 min)
- Smoke tests fail in CI but pass locally

**Causes**:
- Heavy test lacking `@pytest.mark.validation`
- Statistical test in smoke suite

**Mitigation**:
- CI guard test (`tests/test_marker_enforcement.py`)
- Code review checklist (see `docs/TEST_POLICY.md`)

### Loss of Criticality Control
**Symptoms**:
- `sigma` diverges from target
- Network enters supercritical (continuous bursting) or silent state

**Causes**:
- PID gains too aggressive/conservative
- Temperature modulation conflicts with sigma controller
- External drive overpowers homeostasis

**Mitigation**:
- Gradual PID tuning with small steps
- Temperature schedule coordinated with control loops
- Bounds on control signal magnitude

---

## Operating Envelope

### Recommended Parameter Ranges
```python
# Network
N = [100, 10000]           # Neuron count
p_conn = [0.01, 0.2]       # Connection probability
dt = [0.05, 0.5] ms        # Timestep (0.1 ms nominal)

# Temperature
T = [0.1, 2.0]             # Modulation factor
cooling_rate ≤ 0.5 / s     # Avoid rapid quenches

# Sigma Control
sigma_target = [0.5, 1.5]  # Normalized stdev
k_p = [0.0, 0.5]           # Proportional gain
update_interval = 10 ms    # Controller sample period

# Plasticity
tau_cons = [1000, 10000] ms   # Consolidation timescale
A_plus, A_minus = [0.0, 0.01]  # STDP amplitude
```

### Performance Budgets
- **Smoke Test Suite**: <30 seconds
- **Validation Suite**: <5 minutes
- **Single Simulation** (N=1000, t=2000 ms): ~2 seconds (CPU)
- **Memory**: O(N²) connectivity, O(N) state vectors

### Scaling Limits
- **Small (N<500)**: Exact connectivity matrix, direct simulation
- **Medium (N=500–5000)**: Sparse connectivity, vectorized updates
- **Large (N>5000)**: Consider JAX/Torch backends (optional)

---

## Invariants & Sanity Checks

### Runtime Assertions (Debug Mode)
```python
assert np.all(np.isfinite(V)), "Membrane potentials diverged"
assert np.all(W >= 0), "Negative synaptic weights"
assert 0 <= sigma <= 10, "Unrealistic membrane potential spread"
```

### Post-Simulation Validation
```python
# Spike count sanity
assert spike_count > 0, "Silent network (no spikes)"
assert spike_count < N * T_sim * 200, "Pathological firing rate"

# Energy budget
total_spikes = np.sum(spike_history)
assert total_spikes < 1e9, "Explosion detected"
```

### Determinism Check
```python
# Golden hash (see tests/test_golden_hash.py)
digest = compute_run_hash(seed=42, params=default_params)
assert digest == EXPECTED_HASH, "Non-deterministic replay"
```

---

## References

- **Source Code**: `src/bnsyn/`
  - `control.py`: Sigma controller
  - `temperature/`: Temperature schedules
  - `sim/network.py`: Main simulation loop
  - `rng.py`: Deterministic RNG management

- **Tests**: `tests/`
  - `test_determinism.py`: Replay validation
  - `test_golden_hash.py`: Hash-based determinism lock
  - `test_properties_*.py`: Invariant checks

- **Documentation**:
  - `docs/SPEC.md`: System specification
  - `docs/ARCHITECTURE.md`: Component design
  - `docs/TEST_POLICY.md`: Test tier definitions

---

## Revision History

| Version | Date       | Changes                          |
|---------|------------|----------------------------------|
| 1.0     | 2026-01-26 | Initial control contract         |

# Coq Formal Proofs for BNsyn

This directory contains Coq proof obligations and formal proofs for the BNsyn thermostated bio-AI system.

## Status

**🟢 ACTIVE - Initial proofs implemented**

This directory contains formal proofs in Coq for critical BNsyn properties. Currently implemented:
- `BNsyn_Sigma.v`: Sigma bounds preservation proofs (COMPLETE ✅)

Additional proof obligations are documented below for future work.

## Implemented Proofs

### BNsyn_Sigma.v - Sigma Bounds Preservation

**Status**: ✅ Complete and verified

**Theorems**:
1. `clamp_preserves_bounds`: General clamp function preserves min/max bounds
2. `sigma_clamp_preserves_bounds`: Sigma clamping preserves [σ_min, σ_max] bounds  
3. `sigma_update_bounded`: Any sigma update using clamp stays in bounds
4. `clamp_idempotent`: Clamp operation is idempotent

**Compiling locally**:
```bash
cd specs/coq
coqc BNsyn_Sigma.v
```

**CI Integration**: `.github/workflows/formal-coq.yml` runs nightly

## Purpose

While TLA+ model checking explores a finite state space to find invariant violations, Coq provides:
- **Theorem proving**: Mechanically verified proofs that hold for all possible inputs
- **Functional correctness**: Prove that implementations match specifications
- **Mathematical rigor**: Establish properties through constructive proofs

## Proof Obligations

The following properties should be formally proven in Coq:

### PO-1: Temperature Schedule Correctness

**Theorem**: The geometric temperature schedule converges to Tmin and is monotonically decreasing.

```coq
Theorem temperature_convergence :
  forall (T0 Tmin alpha : R) (n : nat),
    0 < Tmin < T0 ->
    0 < alpha < 1 ->
    exists (N : nat),
      forall (m : nat), m >= N ->
        abs (temperature_at_step T0 alpha m - Tmin) < epsilon.
```

```coq
Theorem temperature_monotone :
  forall (T0 Tmin alpha : R) (n : nat),
    0 < Tmin < T0 ->
    0 < alpha <= 1 ->
    temperature_at_step T0 alpha n > Tmin ->
    temperature_at_step T0 alpha (S n) <= temperature_at_step T0 alpha n.
```

### PO-2: Plasticity Gate Bounds

**Theorem**: The plasticity gate function always produces values in [0, 1].

```coq
Theorem gate_sigmoid_bounds :
  forall (T Tc tau : R),
    tau > 0 ->
    0 <= gate_sigmoid T Tc tau <= 1.
```

**Theorem**: The gate is continuous and monotonically increasing with temperature.

```coq
Theorem gate_sigmoid_monotone :
  forall (T1 T2 Tc tau : R),
    tau > 0 ->
    T1 < T2 ->
    gate_sigmoid T1 Tc tau < gate_sigmoid T2 Tc tau.
```

### PO-3: Sigma Bounds Preservation

**Theorem**: If sigma starts in bounds and updates preserve bounds, it remains in bounds.

```coq
Theorem sigma_clamp_preservation :
  forall (sigma sigma' sigma_min sigma_max : R),
    sigma_min <= sigma <= sigma_max ->
    sigma' = clamp sigma_min sigma_max (update_sigma sigma) ->
    sigma_min <= sigma' <= sigma_max.
```

### PO-4: Phase State Machine Well-Formedness

**Theorem**: Phase transitions form a valid state machine without invalid transitions.

```coq
Inductive Phase : Type :=
  | Active : Phase
  | Consolidating : Phase
  | Cooled : Phase.

Inductive valid_transition : Phase -> Phase -> Prop :=
  | Active_to_Active : valid_transition Active Active
  | Active_to_Consolidating : valid_transition Active Consolidating
  | Consolidating_to_Consolidating : valid_transition Consolidating Consolidating
  | Consolidating_to_Cooled : valid_transition Consolidating Cooled
  | Cooled_to_Cooled : valid_transition Cooled Cooled.

Theorem no_invalid_transitions :
  forall (p1 p2 : Phase),
    phase_step p1 p2 ->
    valid_transition p1 p2.
```

### PO-5: Determinism

**Theorem**: Given the same initial state and random seed, the system produces identical outputs.

```coq
Theorem simulation_deterministic :
  forall (state1 state2 : SystemState) (seed : nat) (steps : nat),
    state1 = state2 ->
    run_simulation state1 seed steps = run_simulation state2 seed steps.
```

### PO-6: Energy Conservation

**Theorem**: Total energy (kinetic + potential) is conserved in the absence of external input.

```coq
Theorem energy_conservation :
  forall (state : NetworkState) (dt : R),
    no_external_input state ->
    abs (total_energy (step_network state dt) - total_energy state) < epsilon.
```

### PO-7: Numerical Stability

**Theorem**: For sufficiently small dt, the numerical integration is stable (bounded error).

```coq
Theorem integration_stability :
  forall (state : NeuronState) (dt : R),
    0 < dt < dt_max ->
    bounded (step_neuron state dt).
```

## Implementation Roadmap

### Phase 1: Setup
- [ ] Define Coq environment and dependencies
- [ ] Create base type definitions matching Python implementation
- [ ] Establish equivalence between Coq and Python types

### Phase 2: Core Proofs
- [ ] Prove PO-1 (Temperature schedule correctness)
- [ ] Prove PO-2 (Gate bounds)
- [ ] Prove PO-3 (Sigma preservation)

### Phase 3: System Properties
- [ ] Prove PO-4 (State machine well-formedness)
- [ ] Prove PO-5 (Determinism)

### Phase 4: Advanced Properties
- [ ] Prove PO-6 (Energy conservation)
- [ ] Prove PO-7 (Numerical stability)

## Development Environment

### Installing Coq

```bash
# Using opam (OCaml package manager)
opam install coq coq-ide

# Or using system package manager
sudo apt-get install coq coqide  # Debian/Ubuntu
brew install coq                  # macOS
```

### Recommended Coq Version

- Coq 8.15 or later
- CoqIDE or Proof General for interactive development

### Required Libraries

```bash
opam install coq-mathcomp-ssreflect
opam install coq-mathcomp-algebra
opam install coq-coquelicot  # Real analysis
```

## Resources

### Coq Documentation
- **Official Coq Manual**: https://coq.inria.fr/refman/
- **Software Foundations**: https://softwarefoundations.cis.upenn.edu/
- **Programs and Proofs**: https://ilyasergey.net/pnp/

### Relevant Coq Projects
- **CompCert**: Verified C compiler
- **Flocq**: Floating-point arithmetic formalization
- **Coquelicot**: Real analysis library

### BNsyn Context
- See `docs/SPEC.md` for system specification
- See `specs/tla/BNsyn.tla` for TLA+ model
- See `src/bnsyn/` for Python reference implementation

## Contributing

When implementing proofs:

1. Start with the simplest properties (PO-2: Gate bounds)
2. Build up to compositional properties
3. Use Coq's standard library and mathcomp when possible
4. Document proof structure and key lemmas
5. Extract executable OCaml code to validate against Python implementation

## Future Work

- Formalize the complete AdEx neuron dynamics
- Prove convergence properties of STDP learning rules
- Establish criticality theorems for branching ratio
- Verify error bounds for numerical integration schemes
- Prove safety properties for the consolidated memory system

## Integration with CI/CD

Once proofs are implemented, add a workflow:
- `.github/workflows/formal-coq.yml`
- Compile all Coq files (`coqc *.v`)
- Verify proofs pass
- Extract and test generated OCaml code

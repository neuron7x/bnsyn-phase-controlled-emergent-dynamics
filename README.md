# BN-Syn Thermostated Bio-AI System

A biologically-inspired neural simulation toolkit implementing:

1) AdEx neurons + conductance synapses (AMPA/NMDA/GABA_A)  
2) Three-factor plasticity (eligibility traces × neuromodulator)  
3) Criticality control via branching parameter σ with gain adaptation  
4) Temperature-gated exploration ↔ consolidation with dual-weight synapses  
5) Deterministic, Δt-invariant validation harnesses

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest -m "not validation"
```

Run a minimal demo simulation:

```bash
bnsyn demo --steps 2000 --dt-ms 0.1 --seed 42
```

Run Δt invariance harness:

```bash
bnsyn dtcheck --dt-ms 0.1 --dt2-ms 0.05 --steps 2000 --seed 42
```

## Repository map

- `docs/SPEC.md` — formal 12-component specification (equations, calibration, failure envelopes)
- `src/bnsyn/` — reference implementation
- `tests/` — smoke + validation tests (CI excludes `-m validation`)

## Determinism contract

- single entrypoint seeding: `bnsyn.rng.seed_all(seed)`
- all stochastic terms use `numpy.random.Generator` passed explicitly (no hidden global RNG)
- Δt-invariance checks compare dt vs dt/2 runs under the same seed

## Sources

Primary references are listed in `docs/SPEC.md` and are aligned to the equations implemented here.

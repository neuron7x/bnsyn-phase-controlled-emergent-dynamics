# BN-Syn Thermostated Bio-AI System

A biologically-inspired neural simulation toolkit implementing:

1) AdEx neurons + conductance synapses (AMPA/NMDA/GABA_A)  
2) Three-factor plasticity (eligibility traces × neuromodulator)  
3) Criticality control via branching parameter σ with gain adaptation  
4) Temperature-gated exploration ↔ consolidation with dual-weight synapses  
5) Deterministic, Δt-invariant validation harnesses

---

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

---

## Repository Map

| Path | Purpose |
|------|---------|
| [`docs/INDEX.md`](docs/INDEX.md) | **Navigation hub** — start here for all documentation |
| [`docs/SPEC.md`](docs/SPEC.md) | Formal 12-component specification |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Architecture ↔ evidence crosswalk |
| [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md) | Governance entry (SSOT, claims, validators) |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Determinism protocol |
| [`docs/INVENTORY.md`](docs/INVENTORY.md) | Governed path inventory |
| [`src/bnsyn/`](src/bnsyn/) | Reference implementation |
| [`tests/`](tests/) | Smoke + validation tests |
| [`bibliography/`](bibliography/) | SSOT bibliography artifacts |
| [`claims/`](claims/) | Evidence ledger |
| [`scripts/`](scripts/) | SSOT validators |

---

## SSOT Gates (Run These)

```bash
# Validate SSOT closure
python scripts/validate_bibliography.py
python scripts/validate_claims.py
python scripts/scan_normative_tags.py

# Or use Makefile
make ssot
```

---

## Test Commands

```bash
# Smoke tests (fast, CI default)
pytest -m "not validation"
make test-smoke

# Validation tests (slow, statistical)
pytest -m validation
make test-validation

# Full local CI check
make ci-local
```

---

## Determinism Contract

- **Single entrypoint seeding**: `bnsyn.rng.seed_all(seed)`
- **Explicit RNG**: all stochastic terms use `numpy.random.Generator` passed explicitly
- **Δt-invariance**: checks compare dt vs dt/2 runs under the same seed

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for full protocol.

---

## Governance & Evidence

- **SSOT Policy**: [`docs/SSOT.md`](docs/SSOT.md) (summary) + [`docs/SSOT_RULES.md`](docs/SSOT_RULES.md) (authoritative rules)
- **Claims Ledger**: [`claims/claims.yml`](claims/claims.yml)
- **Bibliography**: [`bibliography/bnsyn.bib`](bibliography/bnsyn.bib)
- **Validators**: `scripts/validate_bibliography.py`, `scripts/validate_claims.py`, `scripts/scan_normative_tags.py`

CI enforces SSOT gates on every PR. See [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md) for details.

---

## Sources

Primary references are listed in [`docs/SPEC.md`](docs/SPEC.md) and [`docs/BIBLIOGRAPHY.md`](docs/BIBLIOGRAPHY.md).
All normative claims cite Tier-A peer-reviewed sources with DOI.

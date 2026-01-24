# BN-Syn Thermostated Bio-AI System

BN-Syn is the deterministic reference implementation of the BN-Syn Thermostated Bio-AI System defined by the specification and governance artifacts in this repository.

[![ci-pr](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-pr.yml/badge.svg?branch=main)](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-pr.yml)
[![ci-validation](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-validation.yml/badge.svg?branch=main)](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-validation.yml)
[![codeql](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/codeql.yml)
[![codecov](https://codecov.io/gh/neuron7x/bnsyn-phase-controlled-emergent-dynamics/branch/main/graph/badge.svg?token=CODECOV_TOKEN)](https://codecov.io/gh/neuron7x/bnsyn-phase-controlled-emergent-dynamics)
[![ci-pr-atomic](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-pr-atomic.yml/badge.svg?branch=main)](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-pr-atomic.yml)

## Start here

- **Documentation hub:** [`docs/INDEX.md`](docs/INDEX.md)
- **Formal spec:** [`docs/SPEC.md`](docs/SPEC.md)
- **Architecture + evidence crosswalk:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **SSOT policy:** [`docs/SSOT.md`](docs/SSOT.md)
- **CI contract:** [`docs/CI_GATES.md`](docs/CI_GATES.md)
- **Governance entrypoint:** [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md)
- **Bibliography docs:** [`docs/BIBLIOGRAPHY.md`](docs/BIBLIOGRAPHY.md) and [`bibliography/`](bibliography/)
- **Claims ledger:** [`claims/claims.yml`](claims/claims.yml)
- **Validators & tooling:** [`scripts/`](scripts/)
- **Repository policies:** [`SECURITY.md`](SECURITY.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`LICENSE`](LICENSE), [`CITATION.cff`](CITATION.cff)

## Repository contract

**Determinism rules (repo contract):**
- All randomness flows through `numpy.random.Generator` created by `bnsyn.rng.seed_all(seed)`.
- No hidden global RNGs inside modules.
- All stochastic updates use √dt scaling for noise terms (when present).
- Δt-invariance checks compare dt vs dt/2 against reference tolerances.

**Test partition:**
- Smoke tests: `pytest -m "not validation"`
- Validation tests: `pytest -m validation`

**SSOT gates:**
- `python scripts/validate_bibliography.py`
- `python scripts/validate_claims.py`
- `python scripts/scan_normative_tags.py`
- `python scripts/scan_governed_docs.py`

## Quickstart

```bash
python -m pip install -e ".[dev]"
```

## PR Quality Standards

All PRs must pass:
- ✅ Pre-commit hooks (ruff, mypy, pylint, pydocstyle, pytest-smoke, coverage)
- ✅ Unit tests with ≥85% coverage
- ✅ Determinism checks (identical seed = identical results)
- ✅ SSOT gates (bibliography, claims, normative tags)
- ✅ Security audits (gitleaks, pip-audit, bandit)
- ✅ Branch protection rules (all required checks must pass)

## Local Development

```bash
make dev-setup
make check
make test
make coverage
```

## Testing in Docker

```bash
docker build -t bnsyn-dev .
docker run bnsyn-dev
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full developer workflow.

```bash
python scripts/validate_bibliography.py
python scripts/validate_claims.py
python scripts/scan_normative_tags.py
python scripts/scan_governed_docs.py
```

```bash
pytest -m "not validation"
```

```bash
pytest -m validation
```

## CI on Pull Requests

CI requirements and exact commands are defined in [`docs/CI_GATES.md`](docs/CI_GATES.md). PRs run the following jobs:

- **ssot** (SSOT validation)
- **quality** (ruff format/lint + mypy)
- **build** (package build + import)
- **tests-smoke** (pytest -m "not validation")
- **gitleaks** (secret scanning)
- **pip-audit** (dependency vulnerability audit)

```mermaid
flowchart TD
  PR[Pull Request] --> ssot[ssot]
  PR --> quality[quality]
  PR --> build[build]
  PR --> tests_smoke[tests-smoke]
  PR --> gitleaks[gitleaks]
  PR --> pip_audit[pip-audit]
```

## Architecture at a glance

```mermaid
flowchart TD
  micro[Micro: AdEx + conductance synapses] --> meso[Meso: three-factor plasticity]
  meso --> macro[Macro: criticality control (σ tracking + gain)]
  macro --> meta[Meta: temperature schedule + gating]
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/SPEC.md`](docs/SPEC.md) for the authoritative architecture and equations.

## Evidence & Bibliography

- **Bibliography overview:** [`docs/BIBLIOGRAPHY.md`](docs/BIBLIOGRAPHY.md)
- **Source artifacts:** [`bibliography/bnsyn.bib`](bibliography/bnsyn.bib), [`bibliography/mapping.yml`](bibliography/mapping.yml), [`bibliography/sources.lock`](bibliography/sources.lock)
- **Claims ledger:** [`claims/claims.yml`](claims/claims.yml)
- **Validators:** [`scripts/validate_bibliography.py`](scripts/validate_bibliography.py), [`scripts/validate_claims.py`](scripts/validate_claims.py), [`scripts/scan_normative_tags.py`](scripts/scan_normative_tags.py)

Tier definitions (per SSOT policy):
- **Tier-A** sources are peer-reviewed with DOI and are normative.
- **Tier-S** sources are standards/documentation without DOI and are non-normative.
- **Tier-B** sources are conference/workshop papers and are non-normative.
- **Tier-C** sources are other sources and are non-normative.

## How to cite

See [`CITATION.cff`](CITATION.cff).

## License / Security / Contributing

- License: [`LICENSE`](LICENSE)
- Security policy: [`SECURITY.md`](SECURITY.md)
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)

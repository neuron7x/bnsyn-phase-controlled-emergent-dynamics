<div align="center">
  <img src="docs/assets/hero.svg" alt="BN-Syn project banner" width="100%" />
</div>

# BN-Syn Thermostated Bio-AI System

BN-Syn is an engineering-grade simulation repository for phase-controlled emergent neural dynamics.
It ships code, tests, reproducibility checks, and governance artifacts in one versioned tree.
Use this repo to run deterministic demos, validate invariants, and inspect evidence-backed outputs.
The runtime code lives in `src/bnsyn/`; CI/workflow policy lives in `.github/workflows/`.
Generated artifacts are written to `artifacts/` for local verification.
Default development flow is Make-target driven and lockfile-oriented.
No secrets are required for local demo, fast tests, or reproducibility proof steps.
For onboarding, use exactly one path: `docs/START_HERE.md`.

[![CI PR](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-pr.yml/badge.svg)](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-pr.yml)
[![Atomic PR Gate](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-pr-atomic.yml/badge.svg)](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/ci-pr-atomic.yml)
[![Docs](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/docs.yml/badge.svg)](https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics/actions/workflows/docs.yml)

## Quickstart (canonical)

```bash
make setup
make demo
```

## Canonical links

- Onboarding funnel: [docs/START_HERE.md](docs/START_HERE.md)
- Reproduce proof: [docs/proof/REPRODUCE.md](docs/proof/REPRODUCE.md)
- Contributing workflow: [docs/TESTING.md](docs/TESTING.md)
- Security docs: [docs/SECURITY_GITLEAKS.md](docs/SECURITY_GITLEAKS.md)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Maintainers / Repo Contract

```bash
make quickstart-smoke
python -m pip install -e .
python -m bnsyn --help
bnsyn demo --steps 120 --dt-ms 0.1 --seed 123 --N 32
```


## Test commands

```bash
make test-gate
make test
make test-all
```

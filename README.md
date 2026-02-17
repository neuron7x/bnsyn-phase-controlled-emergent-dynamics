# BN-Syn Thermostated Bio-AI System

Deterministic research-grade simulator and governance repository for phase-controlled emergent dynamics.

## Status
- **Maturity:** research-grade / pre-production.
- **Missing product/ops layers:** deployment SLOs, runtime service API compatibility policy, and formal release semver governance are not fully defined in SSOT docs.

## Canonical onboarding
- [docs/STATUS.md](docs/STATUS.md)
- [docs/INDEX.md](docs/INDEX.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/PROJECT_SURFACES.md](docs/PROJECT_SURFACES.md)
- [docs/ENFORCEMENT_MATRIX.md](docs/ENFORCEMENT_MATRIX.md)

## Canonical commands
```bash
python -m pip install -e "."
python -m bnsyn --help
python -m pytest -m "not validation" -q
ruff check .
pylint src/bnsyn
mypy src --strict --config-file pyproject.toml
python -m build
python -m scripts.validate_traceability
python -m scripts.discover_public_surfaces
python -m scripts.check_internal_links
```

## Public surface summary
- CLI entrypoint: `bnsyn` (`pyproject.toml`)
- Python package exports: `src/bnsyn/__init__.py::__all__`
- Artifact schemas: `schemas/*.json`

For stability labels and compatibility commitments, see [docs/API_STABILITY.md](docs/API_STABILITY.md) and [docs/VERSIONING.md](docs/VERSIONING.md).

## Quickstart contract
```bash
make quickstart-smoke
python -m pip install -e .
python -m bnsyn --help
bnsyn demo --steps 120 --dt-ms 0.1 --seed 123 --N 32
```

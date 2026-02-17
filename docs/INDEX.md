# Documentation Index (Authoritative Hub)

Single navigation hub for repository-facing docs.

## Start
1. [README.md](../README.md)
2. [START_HERE.md](START_HERE.md)
3. [ARCHITECTURE.md](ARCHITECTURE.md)

## Governance and SSOT
- [SSOT.md](SSOT.md)
- [ENFORCEMENT_MATRIX.md](ENFORCEMENT_MATRIX.md)
- [TRACEABILITY.md](TRACEABILITY.md)
- [PROJECT_SURFACES.md](PROJECT_SURFACES.md)
- [API_STABILITY.md](API_STABILITY.md)
- [VERSIONING.md](VERSIONING.md)
- [DETERMINISM.md](DETERMINISM.md)

## Operational docs
- [INPUTS.md](INPUTS.md)
- [ASSUMPTIONS.md](ASSUMPTIONS.md)
- [RISKS.md](RISKS.md)
- [DOC_DEBT.md](DOC_DEBT.md)
- [MAINTENANCE.md](MAINTENANCE.md)
- [TESTING.md](TESTING.md)
- [CI_GATES.md](CI_GATES.md)

## Canonical commands
```bash
python -m pytest -m "not validation" -q
ruff check .
pylint src/bnsyn
mypy src --strict --config-file pyproject.toml
python -m build
python -m scripts.validate_traceability
python -m scripts.discover_public_surfaces
```

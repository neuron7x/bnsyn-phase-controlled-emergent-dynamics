# Documentation Index (Authoritative Hub)

Single authoritative navigation hub for repository-facing documentation.

## Start here

1. [README.md](../README.md)
2. [STATUS.md](STATUS.md)
3. [ARCHITECTURE.md](ARCHITECTURE.md)

## Core architecture + governance surfaces

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [PROJECT_SURFACES.md](PROJECT_SURFACES.md)
- [ENFORCEMENT_MATRIX.md](ENFORCEMENT_MATRIX.md)
- [TRACEABILITY.md](TRACEABILITY.md)
- [SSOT.md](SSOT.md)

## Operational and quality docs

- [TESTING.md](TESTING.md)
- [CI_GATES.md](CI_GATES.md)
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- [MAINTENANCE.md](MAINTENANCE.md)
- [DOC_DEBT.md](DOC_DEBT.md)

## Canonical commands

```bash
python -m pytest -m "not validation" -q
ruff check .
pylint src/bnsyn
mypy src --strict --config-file pyproject.toml
python -m build
python -m scripts.validate_traceability
python -m scripts.discover_public_surfaces --check
python -m scripts.check_internal_links
```

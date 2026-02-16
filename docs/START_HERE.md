# Start Here

Use this page as the canonical onboarding path for new engineers.

## 1) What this repository is

BN-Syn is a deterministic reference implementation of phase-controlled emergent neural dynamics with specification, governance, and evidence artifacts versioned together.

- System specification: [SPEC.md](SPEC.md)
- Architecture crosswalk: [ARCHITECTURE.md](ARCHITECTURE.md)
- API contract: [API_CONTRACT.md](API_CONTRACT.md)

## 2) Local setup

```bash
python -m pip install -e ".[dev]"
```

Optional extras:

- Visualization: `python -m pip install -e ".[dev,viz]"`
- Accelerator experiments: `python -m pip install -e ".[dev,accelerators]"`

## 3) Three canonical commands

```bash
# Run
python -m bnsyn.cli --help

# Test
pytest -q

# Build docs
python -m sphinx -b html docs docs/_build/html
```

## 4) Primary documentation paths

- Concepts & architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Scripts & tools registry: [SCRIPTS/index.md](SCRIPTS/index.md)
- API reference (Sphinx): [api/index.md](api/index.md)
- Testing and validation workflows: [TESTING.md](TESTING.md)
- Troubleshooting: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## 5) Reproducibility and evidence

- Reproducibility envelope: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- CI gates: [CI_GATES.md](CI_GATES.md)
- Evidence coverage: [EVIDENCE_COVERAGE.md](EVIDENCE_COVERAGE.md)


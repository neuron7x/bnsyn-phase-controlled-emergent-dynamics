# START_HERE

This is the canonical onboarding funnel.

## 1) Install

```bash
make setup
```

## 2) Run canonical product demo

```bash
python -m bnsyn demo-product
python -m bnsyn validate-bundle artifacts/canonical_run
```

Base mode artifacts (`bnsyn run --profile canonical --plot`):
- `emergence_plot.png`
- `summary_metrics.json`
- `run_manifest.json`
- `criticality_report.json`
- `avalanche_report.json`
- `phase_space_report.json`

Export-proof additional artifact (`bnsyn run --profile canonical --plot --export-proof`):
- `proof_report.json`

## 3) Run tests

```bash
make test
```

## 4) Reproducibility check

```bash
make reproduce
```

Expected artifacts:
- `artifacts/demo.json`
- `artifacts/demo.sha256`
- `artifacts/reproduce_manifest.json`
- `artifacts/reproducibility_report.json`

## 5) Canonical references

- Proof contract: [docs/CANONICAL_PROOF.md](CANONICAL_PROOF.md)
- Mechanism + architecture: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- Reproducibility details: [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md)

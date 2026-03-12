# Deterministic Quickstart Contract

This quickstart is a runnable contract:

1. bootstrap local Linux environment
2. verify CLI and dependencies
3. run canonical visual proof command
4. verify canonical artifact outputs

## One obvious Linux path (recommended)

```bash
./scripts/bootstrap_local_linux.sh
./scripts/run_canonical_local.sh
```

## Manual install path (equivalent)

```bash
python -m pip install -e ".[plot]"
python -m scripts.local_doctor
python -m bnsyn --help
bnsyn run --profile canonical --plot --export-proof --output artifacts/canonical_run
```

## Makefile helpers

```bash
make local-bootstrap
make local-verify
make local-run
# or do both bootstrap+run:
make local-all
```

## Expected output contract

CLI stdout is a JSON object containing:

- `status` set to `ok`
- `artifact_dir`
- `artifacts` containing:
  - `emergence_plot.png`
  - `summary_metrics.json`
  - `run_manifest.json`
  - `criticality_report.json`
  - `avalanche_report.json`
  - `phase_space_report.json`
  - `proof_report.json` (only with `--export-proof`)

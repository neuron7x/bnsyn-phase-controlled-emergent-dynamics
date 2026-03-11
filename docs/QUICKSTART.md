# Deterministic Quickstart Contract

This quickstart is a runnable contract:

1. install
2. verify CLI
3. run canonical visual proof command
4. verify canonical artifact outputs

## Install

```bash
python -m pip install -e .
```

## Verify CLI

```bash
python -m bnsyn --help
```

## Run canonical proof command

```bash
bnsyn run --profile canonical --plot --export-proof
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

# 🚀 60-Second Quickstart

Get from zero to one deterministic BN-Syn proof run.

## Canonical install path

BN-Syn is source-first in this repository.
Use an editable install from source as the canonical path.

## Supported Python versions

- Python 3.11+ (`requires-python = ">=3.11"`)

## Install

```bash
python -m pip install -e ".[plot]"
```

## Verify CLI

```bash
python -m bnsyn --help
```

## Run canonical proof command

```bash
bnsyn run --profile canonical --plot --export-proof
```

Expected output contract: JSON with `status="ok"` and canonical artifacts list including `criticality_report.json`, `avalanche_report.json`, `phase_space_report.json`, and `proof_report.json`.

## Reproducible smoke target

```bash
make quickstart-smoke
```

## Optional: sleep-stack command discovery

```bash
bnsyn sleep-stack --help
```

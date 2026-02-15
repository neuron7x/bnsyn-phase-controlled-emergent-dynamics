# 🚀 60-Second Quickstart

Get from zero to a deterministic BN-Syn run in about 60 seconds.

## Canonical install path

BN-Syn is currently **source-first** in this repository.
Use an editable install from source as the canonical path.

## Supported Python versions

- Python 3.11+ (`requires-python = ">=3.11"`)

## Install

```bash
python -m pip install -e .
```

## Verify CLI

```bash
python -m bnsyn --help
```

## Run deterministic quickstart demo

```bash
bnsyn demo --steps 120 --dt-ms 0.1 --seed 123 --N 32
```

Expected output contract: JSON with top-level key `"demo"`.

## Optional: sleep-stack command discovery

```bash
bnsyn sleep-stack --help
```

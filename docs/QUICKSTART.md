# Deterministic Quickstart Contract

This quickstart is a runnable contract:

1. install
2. verify CLI
3. run minimal deterministic simulation
4. verify expected output shape

## Install

```bash
python -m pip install -e .
```

## Verify CLI

```bash
python -m bnsyn --help
```

## Run minimal deterministic simulation

```bash
bnsyn demo --steps 120 --dt-ms 0.1 --seed 123 --N 32
```

## Expected output contract

Top-level JSON object with field `demo` and deterministic metric keys:

- `demo.sigma_mean`
- `demo.sigma_std`
- `demo.rate_mean_hz`
- `demo.rate_std`

Expected value ranges:

- `0.0 <= demo.sigma_mean <= 5.0`
- `0.0 <= demo.rate_mean_hz <= 1000.0`

# Emergence experiment (external-current controlled)

This path provides a reproducible way to drive BN-Syn from low/silent activity toward active regimes using controlled `external_current_pA`.

## Runtime contract

- `duration_ms` must be an integer multiple of `dt_ms` (no truncation is performed).
- `N > 0`, `seed > 0`, `dt_ms > 0`, and `external_current_pA` must be finite.
- `emergence-plot` requires the optional visualization dependency (`matplotlib`).

## Canonical single run

```bash
PYTHONPATH=src python -m bnsyn.cli emergence-run \
  --N 500 --dt-ms 0.1 --duration-ms 2000 --seed 42 \
  --external-current-pA 410.0 \
  --out artifacts/emergence/run_410
```

Outputs:
- `artifacts/emergence/run_410/run_42_Iext_410pA.npz` (artifact with stable field contract)
- `artifacts/emergence/run_410/emergence_run_report.json` (machine-readable summary)

## Fixed sweep

```bash
PYTHONPATH=src python -m bnsyn.cli emergence-sweep \
  --N 500 --dt-ms 0.1 --duration-ms 2000 --seed 42 \
  --out artifacts/emergence/sweep
```

Sweep currents are fixed to `[365.0, 380.0, 395.0, 410.0, 450.0]` pA.

Outputs:
- `artifacts/emergence/sweep/emergence_sweep_report.json`
- `artifacts/emergence/sweep/run_42_Iext_<current>pA.npz` for each current

## Plot generation from saved artifact

```bash
PYTHONPATH=src python -m bnsyn.cli emergence-plot \
  --input artifacts/emergence/run_410/run_42_Iext_410pA.npz \
  --output artifacts/emergence/run_410/emergence_410.png
```

The figure includes:
- spike raster
- population rate trace
- sigma trace

## Claim boundary

Supported by this path:
- controlled external current changes simulated activity regime
- run/sweep metrics and plots are reproducible by CLI commands above

Not supported by this path:
- biological equivalence
- claims about consciousness/intelligence
- avalanche-criticality claims beyond reported sigma traces

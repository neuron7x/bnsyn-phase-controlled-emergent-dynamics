# BN-Syn Offline Demo (Deterministic)

This demo runs fully offline with a fixed seed and writes reproducible artifacts.

## Canonical demo command

```bash
bnsyn run --profile canonical --plot --export-proof
```

Outputs:
- `artifacts/canonical_run/emergence_plot.png`
- `artifacts/canonical_run/summary_metrics.json`
- `artifacts/canonical_run/run_manifest.json`

## Alternate output directory

```bash
bnsyn run --profile canonical --plot --export-proof --output results/demo_smoke
```

# BN-Syn Offline Demo (Deterministic)

This demo runs fully offline with a fixed seed and writes reproducible artifacts.

## Canonical demo command

```bash
bnsyn plot
```

Outputs:
- `artifacts/canonical_plot/emergence_plot.png`
- `artifacts/canonical_plot/summary_metrics.json`
- `artifacts/canonical_plot/run_manifest.json`

## Alternate output directory

```bash
bnsyn plot --seed 7 --steps 300 --N 96 --out results/demo_smoke
```

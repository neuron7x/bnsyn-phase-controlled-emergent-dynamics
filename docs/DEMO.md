# BN-Syn Offline Demo (Deterministic)

This demo runs fully offline and deterministically using a fixed seed. It produces
reproducible artifacts under `results/` without network access.

For full conference preparation (release readiness + backup plan), see
[`docs/CONFERENCE_RUNBOOK.md`](CONFERENCE_RUNBOOK.md).

## Release Readiness (Blocking)

```bash
python -m scripts.release_readiness
```

**Expected output:**
- `artifacts/release_readiness.json`
- `artifacts/release_readiness.md`
- Terminal message: `Release readiness: READY`

## Quick Demo (Recommended)

```bash
bnsyn sleep-stack --seed 123 --steps-wake 240 --steps-sleep 180 --out results/demo_rc
```

**Expected outputs**
- `results/demo_rc/manifest.json`
- `results/demo_rc/metrics.json`
- `results/demo_rc/summary.json`
- `figures/demo_rc/summary.png` (if `matplotlib` is installed)

## Minimal Sanity Demo (Fast)

```bash
bnsyn sleep-stack --seed 7 --steps-wake 60 --steps-sleep 40 --out results/demo_smoke
```

This run completes in a few seconds and verifies deterministic execution.

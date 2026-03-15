# BN-Syn Conference Runbook (Release-Ready Demo)

This runbook provides deterministic, offline steps for preparing and presenting the BN-Syn demo.
All commands are designed for reproducibility and auditability.

## Preconditions

- Python 3.11+
- Virtual environment activated
- Repository at a clean commit

## 1) Install Dependencies

```bash
pip install -e ".[dev,viz]"
```

## 2) Release Readiness Report (Blocking)

```bash
python -m scripts.release_readiness
```

**Expected output:**
- `artifacts/release_readiness.json`
- `artifacts/release_readiness.md`
- Terminal message: `Release readiness: READY`

If the report is `BLOCKED`, resolve missing files before proceeding.

## 3) Deterministic Demo Run (Primary)

```bash
python -m bnsyn demo-product --output artifacts/canonical_run
python -m bnsyn validate-bundle artifacts/canonical_run
```

**Expected outputs**
- `artifacts/canonical_run/index.html`
- `artifacts/canonical_run/emergence_plot.png`
- `artifacts/canonical_run/summary_metrics.json`
- `artifacts/canonical_run/proof_report.json`
## 4) Fast Sanity Demo (Backup)

```bash
python -m bnsyn demo-product --output results/demo_smoke
python -m bnsyn validate-bundle results/demo_smoke
```

**Expected outputs**
- `results/demo_smoke/index.html`
- `results/demo_smoke/emergence_plot.png`
- `results/demo_smoke/proof_report.json`
## 5) Determinism Spot-Check (Optional)

Re-run the primary demo into a second directory and confirm validation still passes:

```bash
python -m bnsyn demo-product --output results/demo_rc_repeat
python -m bnsyn validate-bundle results/demo_rc_repeat
```

Compare `artifacts/canonical_run/summary_metrics.json` and `results/demo_rc_repeat/summary_metrics.json`.
## 6) Conference Presentation Notes

- Avoid network access during the demo; all artifacts are generated locally.
- Open `index.html` first, then `emergence_plot.png`, then `proof_report.json`.
- `python -m bnsyn validate-bundle <artifact_dir>` must pass before presentation.
- Use the deterministic seed (`123`) for repeatable visuals.
## 7) Post-Demo Cleanup (Optional)

```bash
rm -rf artifacts/canonical_run results/demo_rc_repeat results/demo_smoke
```
```bash
rm -rf results/demo_rc results/demo_rc_repeat results/demo_smoke figures/demo_rc
```

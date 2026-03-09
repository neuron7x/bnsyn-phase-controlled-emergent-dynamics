# Proof: Reproduce

Primary visual proof command:

```bash
bnsyn plot
```

Expected canonical artifacts:

- `artifacts/canonical_plot/emergence_plot.png`
- `artifacts/canonical_plot/summary_metrics.json`
- `artifacts/canonical_plot/run_manifest.json`

---

Canonical command:

```bash
make reproduce
```

This command must produce:

- `artifacts/demo.json`
- `artifacts/demo.sha256`
- `artifacts/reproduce_manifest.json`
- `artifacts/reproducibility_report.json`

Validation rule:
- `artifacts/reproducibility_report.json` must contain `"status": "pass"` for the configured artifact checks.
- `artifacts/demo.sha256` must match the `sha256` value in `artifacts/reproduce_manifest.json`.

# Canonical Emergence Proof Path

`bnsyn run --profile canonical --plot --export-proof` is the canonical command for a single reproducible BN-Syn emergence proof run.

## Command

```bash
bnsyn run --profile canonical --plot --export-proof
```

Optional controls:

```bash
bnsyn run --profile canonical --plot --export-proof --output artifacts/canonical_run
```

## Artifact contract (required)

The command writes exactly this proof bundle to the output directory:

- `emergence_plot.png` — primary canonical emergence visual, a composite image built from spike raster activity and population rate dynamics.
- `summary_metrics.json` — numeric summary for the run.
- `run_manifest.json` — reproducibility manifest including command metadata and artifact hashes for external artifacts; self-entry uses sentinel `"self-unhashed"` to avoid false self-hash claims.
- `criticality_report.json` — machine-readable criticality metrics derived from the canonical run traces.

## Mechanism narrative (research-facing)

BN-Syn models a recurrent spiking network where:

- **AdEx neuron dynamics** provide biologically motivated membrane and adaptation behavior.
- **STDP synapses** implement timing-dependent weight updates; memory/consolidation modules build on these weight dynamics.
- **Criticality control** tracks branching/sigma behavior and adjusts global excitability.

The canonical proof path is intentionally one-run and one-command so external reviewers can inspect concrete behavior without repository archaeology.

## Reproducibility

Reproducibility is enforced through explicit seed, deterministic simulation path, and manifest hashing.
Use `run_manifest.json` + `criticality_report.json` + `summary_metrics.json` to compare repeated runs under same parameters.

## Interpretation layer and limits

- **Directly measured signals:** spike events per step, sigma, active-fraction coherence, spike rate.
- **Derived metrics:** means/peaks and event counts in `summary_metrics.json`.
- **Interpretation layer:** evidence of emergent network dynamics under this model implementation and parameterization.
- **Out of scope / unsupported claims:** cognition, consciousness, AGI/ASI capability, or biological equivalence claims not directly validated by repository experiments.

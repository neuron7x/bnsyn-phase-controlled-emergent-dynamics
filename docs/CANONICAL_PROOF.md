# Canonical Emergence Proof Path

`bnsyn plot` is the canonical command for a single reproducible BN-Syn emergence proof run.

## Command

```bash
bnsyn plot
```

Optional controls:

```bash
bnsyn plot --seed 123 --steps 500 --N 128 --dt-ms 0.5 --backend reference --out artifacts/canonical_plot
```

## Artifact contract (required)

The command writes exactly this proof bundle to the output directory:

- `emergence_plot.png` — four-panel visual evidence:
  - spike raster
  - criticality sigma trace
  - synchronization/coherence trace (active neuron fraction)
  - population activity (spike rate)
- `summary_metrics.json` — numeric summary for the run.
- `run_manifest.json` — reproducibility manifest including command metadata and artifact hashes.

## Mechanism narrative (research-facing)

BN-Syn models a recurrent spiking network where:

- **AdEx neuron dynamics** provide biologically motivated membrane and adaptation behavior.
- **STDP synapses** implement timing-dependent weight updates; memory/consolidation modules build on these weight dynamics.
- **Criticality control** tracks branching/sigma behavior and adjusts global excitability.

The canonical proof path is intentionally one-run and one-command so external reviewers can inspect concrete behavior without repository archaeology.

## Reproducibility

Reproducibility is enforced through explicit seed, deterministic simulation path, and manifest hashing.
Use `run_manifest.json` + `summary_metrics.json` to compare repeated runs under same parameters.

## Interpretation layer and limits

- **Directly measured signals:** spike events per step, sigma, active-fraction coherence, spike rate.
- **Derived metrics:** means/peaks and event counts in `summary_metrics.json`.
- **Interpretation layer:** evidence of emergent network dynamics under this model implementation and parameterization.
- **Out of scope / unsupported claims:** cognition, consciousness, AGI/ASI capability, or biological equivalence claims not directly validated by repository experiments.

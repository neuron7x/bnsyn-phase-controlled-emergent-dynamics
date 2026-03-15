# Canonical Emergence Proof Path

`bnsyn run --profile canonical --plot --export-proof` is the canonical command for a single reproducible BN-Syn emergence proof run.

## Authoritative contract source (SSOT)

The **only authoritative machine-readable canonical proof contract** is:

- `src/bnsyn/resources/ci/canonical_proof_contract.json`

All code paths and CI checks must derive from or validate against this JSON contract. This document is explanatory and non-authoritative for contract details.

## Command

```bash
bnsyn run --profile canonical --plot --export-proof
```

Optional controls:

```bash
bnsyn run --profile canonical --plot --export-proof --output artifacts/canonical_run
```

## Artifact contract (overview)

For the exact required/optional artifact set, schema bindings, invariants, and versioned fields, use `src/bnsyn/resources/ci/canonical_proof_contract.json`.

High-level canonical outputs include:

- `emergence_plot.png`
- `summary_metrics.json`
- `criticality_report.json`
- `avalanche_report.json`
- `phase_space_report.json`
- `population_rate_trace.npy`
- `sigma_trace.npy`
- `coherence_trace.npy`
- `phase_space_rate_sigma.png`
- `phase_space_rate_coherence.png`
- `phase_space_activity_map.png`
- `avalanche_fit_report.json`
- `robustness_report.json`
- `envelope_report.json`
- `run_manifest.json`
- `proof_report.json` (export-proof mode)

## Mechanism narrative (research-facing)

BN-Syn models a recurrent spiking network where:

- **AdEx neuron dynamics** provide biologically motivated membrane and adaptation behavior.
- **STDP synapses** implement timing-dependent weight updates; memory/consolidation modules build on these weight dynamics.
- **Criticality control** tracks branching/sigma behavior and adjusts global excitability.

The canonical proof path is intentionally one-run and one-command so external reviewers can inspect concrete behavior without repository archaeology.

## Reproducibility

Reproducibility is enforced through explicit seed, deterministic simulation path, manifest hashing, and a fixed 10-seed admissibility-band check.
Use `run_manifest.json` + `criticality_report.json` + `avalanche_report.json` + `phase_space_report.json` + `summary_metrics.json` to compare repeated runs under same parameters.

## Interpretation layer and limits

- **Directly measured signals:** spike events per step, sigma, active-fraction coherence, spike rate.
- **Derived metrics:** means/peaks and event counts in `summary_metrics.json`.
- **Interpretation layer:** evidence of emergent network dynamics under this model implementation and parameterization.
- **Out of scope / unsupported claims:** cognition, consciousness, AGI/ASI capability, or biological equivalence claims not directly validated by repository experiments.

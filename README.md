# BN-Syn Thermostated Bio-AI System

BN-Syn is a deterministic simulation repository for phase-controlled emergent neural dynamics with AdEx neurons, STDP plasticity, and criticality control.

## Install

```bash
python -m pip install .
```

## Run

```bash
bnsyn demo-product
```

## Inspect

Open `artifacts/canonical_run/index.html` for the human-readable report, and inspect `artifacts/canonical_run/product_summary.json` for machine-readable status.

## Validate

```bash
bnsyn validate-bundle artifacts/canonical_run
```

Canonical proof command remains:

```bash
bnsyn run --profile canonical --plot --export-proof
```

## Canonical Project Vectors (Permanent)

- **V1 — Result:** [NORMATIVE][CLM-0001] one canonical proof command, `bnsyn run --profile canonical --plot --export-proof`, must generate visual and metrics evidence of emergent network dynamics.
- **V2 — Narrative:** [NORMATIVE][CLM-0002] repository documentation must explain mechanism, measurements, and reproducibility for technical research readers.
- **V3 — Audience:** [NORMATIVE][CLM-0003] repository surfaces must stay runnable and inspectable for AI lab, neuroscience grant, and technical investor diligence.

All contributor work is expected to strengthen these vectors and avoid drift.

## Canonical proof path (single command)

```bash
bnsyn run --profile canonical --plot --export-proof
```

Base canonical artifact contract (`bnsyn run --profile canonical --plot`):
- `emergence_plot.png`
- `summary_metrics.json`
- `run_manifest.json`
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

Export-proof augmented artifact contract (`bnsyn run --profile canonical --plot --export-proof`):
- all base artifacts
- `proof_report.json`

This is the primary buyer/reviewer command path.

## Interpretation and claim boundaries

Supported from the canonical proof bundle:
- direct measured traces from one run: spike raster events, sigma trace, active-fraction coherence trace, spike-rate trace
- derived summary statistics in `summary_metrics.json`
- reproducibility metadata and artifact hashes in `run_manifest.json`
- `criticality_report.json`
- `avalanche_report.json`
- `phase_space_report.json`

Not supported from this proof command alone:
- biological equivalence to in vivo neural tissue
- claims about cognition, consciousness, or AGI-level capability
- generalization claims beyond tested parameter settings and implemented model scope

## Canonical user path (clone -> install -> run -> inspect)

```bash
git clone https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics.git
cd bnsyn-phase-controlled-emergent-dynamics
python3 -m venv .venv
./.venv/bin/python -m pip install -e .
./.venv/bin/python -m bnsyn run --profile canonical --plot --export-proof
```

Inspect:
- `artifacts/canonical_run/emergence_plot.png`
- `artifacts/canonical_run/summary_metrics.json`
- `artifacts/canonical_run/run_manifest.json`
- `artifacts/canonical_run/criticality_report.json`
- `artifacts/canonical_run/avalanche_report.json`
- `artifacts/canonical_run/phase_space_report.json`
- `artifacts/canonical_run/population_rate_trace.npy`
- `artifacts/canonical_run/sigma_trace.npy`
- `artifacts/canonical_run/coherence_trace.npy`
- `artifacts/canonical_run/phase_space_rate_sigma.png`
- `artifacts/canonical_run/phase_space_rate_coherence.png`
- `artifacts/canonical_run/phase_space_activity_map.png`
- `artifacts/canonical_run/avalanche_fit_report.json`
- `artifacts/canonical_run/robustness_report.json`
- `artifacts/canonical_run/envelope_report.json`
- `artifacts/canonical_run/proof_report.json` (when `--export-proof` is enabled)

## Quickstart

```bash
make setup
make demo
make test
```

## Canonical test gate command

```bash
make test-gate
```

## Canonical links

- Onboarding funnel: [docs/START_HERE.md](docs/START_HERE.md)
- Canonical proof contract: [docs/CANONICAL_PROOF.md](docs/CANONICAL_PROOF.md)
- Reproduce proof: [docs/proof/REPRODUCE.md](docs/proof/REPRODUCE.md)
- Contributing workflow: [CONTRIBUTING.md](CONTRIBUTING.md)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Status: [docs/STATUS.md](docs/STATUS.md)

## Maintainers / Repo Contract

```bash
make quickstart-smoke
./.venv/bin/python -m pip install -e .
./.venv/bin/python -m bnsyn --help
./.venv/bin/python -m bnsyn run --profile canonical --plot --export-proof
```

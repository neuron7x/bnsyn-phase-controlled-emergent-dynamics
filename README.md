<div align="center">
  <img src="docs/assets/hero.svg" alt="BN-Syn project banner" width="100%" />
</div>

# BN-Syn Thermostated Bio-AI System

BN-Syn is a deterministic simulation repository for phase-controlled emergent neural dynamics with AdEx neurons, STDP plasticity, and criticality control.

## Canonical Project Vectors (Permanent)

- **V1 — Result:** [NORMATIVE][CLM-0001] one canonical proof command, `bnsyn run --profile canonical --plot --export-proof`, must generate visual and metrics evidence of emergent network dynamics.
- **V2 — Narrative:** [NORMATIVE][CLM-0002] repository documentation must explain mechanism, measurements, and reproducibility for technical research readers.
- **V3 — Audience:** [NORMATIVE][CLM-0003] repository surfaces must stay runnable and inspectable for AI lab, neuroscience grant, and technical investor diligence.

All contributor work is expected to strengthen these vectors and avoid drift.

## Canonical proof path (single command)

```bash
bnsyn run --profile canonical --plot --export-proof
```

Default artifact contract (`artifacts/canonical_run/`):
- `emergence_plot.png`
- `summary_metrics.json`
- `run_manifest.json`
- `criticality_report.json`

This is the primary buyer/reviewer command path.

## Interpretation and claim boundaries

Supported from the canonical proof bundle:
- direct measured traces from one run: spike raster events, sigma trace, active-fraction coherence trace, spike-rate trace
- derived summary statistics in `summary_metrics.json`
- reproducibility metadata and artifact hashes in `run_manifest.json`
- `criticality_report.json`

Not supported from this proof command alone:
- biological equivalence to in vivo neural tissue
- claims about cognition, consciousness, or AGI-level capability
- generalization claims beyond tested parameter settings and implemented model scope

## Canonical user path (clone -> install -> run -> inspect)

```bash
git clone https://github.com/neuron7x/bnsyn-phase-controlled-emergent-dynamics.git
cd bnsyn-phase-controlled-emergent-dynamics
python -m pip install -e .
bnsyn run --profile canonical --plot --export-proof
```

Inspect:
- `artifacts/canonical_run/emergence_plot.png`
- `artifacts/canonical_run/summary_metrics.json`
- `artifacts/canonical_run/run_manifest.json`
- `artifacts/canonical_run/criticality_report.json`

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
python -m pip install -e .
python -m bnsyn --help
bnsyn run --profile canonical --plot --export-proof
```

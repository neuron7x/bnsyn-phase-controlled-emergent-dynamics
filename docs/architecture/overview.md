# Architecture Overview

## Module map

| Module | Purpose | Key entrypoints |
| --- | --- | --- |
| `bnsyn.config` | Typed parameter models for AdEx, synapses, plasticity, criticality, and temperature. | `AdExParams`, `SynapseParams`, `PlasticityParams`, `CriticalityParams`, `TemperatureParams` |
| `bnsyn.rng` | Deterministic RNG seeding and splitting utilities. | `seed_all`, `split`, `RNGPack` |
| `bnsyn.validation` | Input validation for external API boundaries and array validation. | `NetworkValidationConfig`, `validate_state_vector`, `validate_spike_array`, `validate_connectivity_matrix` |
| `bnsyn.sim` | Reference network simulator implementation. | `Network`, `NetworkParams`, `run_simulation` |
| `bnsyn.simulation` | Stable public simulation API surface (re-exports). | `Network`, `NetworkParams`, `run_simulation` |
| `bnsyn.cli` | CLI entrypoints for demos and experiments. | `main` |
| `bnsyn.provenance` | Run manifest and experiment provenance metadata. | `RunManifest`, manifest builders |

## Boundaries and data flow

- **External inputs** enter through the CLI (`bnsyn.cli`) and experiment configuration loaders, then flow into validation helpers in `bnsyn.validation` before reaching the simulator.
- **Simulation stepping** is performed in `bnsyn.sim.network.Network.step`, which integrates neuron, synapse, and criticality updates.
- **Persistence/artifacts** are handled by provenance utilities in `bnsyn.provenance` for run manifests and experiment metadata.
- **Workflow execution paths** run through experiment modules (e.g., `bnsyn.experiments`) that orchestrate simulation runs and aggregates.

## Public API surface

- `bnsyn.simulation` is the stable user-facing entry point for simulation runs and network configuration.
- `bnsyn.rng.seed_all` is the deterministic seed API for all stochastic workflows.
- `bnsyn.config` hosts the parameter models used by public configuration surfaces.

## Related references

- [Architecture ↔ evidence crosswalk](../ARCHITECTURE.md)
- [API contract](../API_CONTRACT.md)
- [Specification](../SPEC.md)

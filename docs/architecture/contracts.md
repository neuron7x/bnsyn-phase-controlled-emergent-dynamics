# Architecture Contracts

## Input validation invariants

- State vectors and connectivity matrices are validated as `float64`, with exact expected shapes, and reject NaN/Inf values.
- Spike arrays are validated as `bool` with exact expected shapes.
- Validation failures raise `ValueError` with explicit diagnostics; type violations raise `TypeError`.

Reference implementation: `bnsyn.validation.inputs`.

## Determinism and seeding

- All stochastic execution paths are seeded via `bnsyn.rng.seed_all`.
- Deterministic behavior is enforced by explicit seeding and reproducible RNG splitting.
- `run_simulation` validates inputs, then seeds RNGs before stepping the network.

Reference implementation: `bnsyn.rng` and `bnsyn.sim.network.run_simulation`.

## Simulation safety bounds

- Network membrane potentials are monitored to remain within configured bounds. If bounds are violated, the simulator raises `RuntimeError` to signal numerical instability.

Reference implementation: `bnsyn.sim.network.Network._raise_if_voltage_out_of_bounds`.

## Failure behavior summary

| Condition | Exception | Origin |
| --- | --- | --- |
| Non-array input for validation helpers | `TypeError` | `bnsyn.validation.inputs._ensure_ndarray` |
| Shape/dtype mismatch for state/connectivity | `ValueError` | `bnsyn.validation.inputs.validate_*` |
| NaN/Inf in state/connectivity | `ValueError` | `bnsyn.validation.inputs.validate_*` |
| Invalid simulation parameters | `ValueError` / `TypeError` | `bnsyn.sim.network.Network` / `run_simulation` |
| Voltage bounds violation | `RuntimeError` | `bnsyn.sim.network.Network._raise_if_voltage_out_of_bounds` |

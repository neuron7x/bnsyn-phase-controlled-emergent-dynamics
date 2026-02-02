# STPA Safety Package

## Scope

The BN-Syn MLSDM safety scope covers deterministic simulation, bounded numerical dynamics, and strict validation of external inputs before they cross API boundaries.

## System-level losses

- **L-001**: Invalid or unsafe simulation outputs due to non-finite numerical state.
- **L-002**: Non-reproducible results that break determinism guarantees.

## Hazards

Hazards are enumerated in [`hazard_log.yml`](hazard_log.yml) and traceable to safety constraints and tests in [`traceability.yml`](traceability.yml).

## Unsafe control actions (UCAs)

- Proceeding with simulation when state or connectivity inputs are non-finite or mismatched.
- Running stochastic simulation paths without deterministic seeding.

## Safety constraints summary

- **SC-001**: Reject non-finite state vectors and connectivity matrices.
- **SC-002**: Require explicit seeding for deterministic execution paths.

## Verification strategy

- Safety constraints are validated with deterministic regression tests and explicit failure-mode checks.
- Machine-checkable schema validation ensures hazard and traceability artifacts remain structurally correct.

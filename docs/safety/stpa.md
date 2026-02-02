# System-Theoretic Process Analysis (STPA)

**Navigation**: [INDEX](../INDEX.md)

## Losses (L)

- **L1**: Loss of deterministic, reproducible simulation outputs.
- **L2**: Loss of safe bounded dynamics (unstable or invalid state propagation).
- **L3**: Loss of memory consistency across model updates.

## Hazards (H)

- **H1**: Invalid external arrays (shape/dtype/NaN) enter the system state.
- **H2**: Network configuration or external current bounds are invalid at runtime.

## Unsafe Control Actions (UCA)

- **UCA1**: Accepting state/connectivity arrays without validating dtype, shape, or NaN.
- **UCA2**: Creating a network with invalid parameters (non-positive N, invalid fractions, non-positive dt).
- **UCA3**: Accepting external current vectors with mismatched shapes.

## Safety Constraints (SC)

- **SC-1**: External arrays **must** be validated for dtype, shape, and NaN presence before use.
- **SC-2**: Network initialization **must** reject invalid parameters and external current shapes.

## Enforcement & Test Mapping

| Safety Constraint | Enforcement (Code) | Tests | Gate |
| --- | --- | --- | --- |
| SC-1 | `bnsyn.validation.inputs` validators | `tests/test_validation_inputs.py` | `pytest -q` |
| SC-2 | `Network.__init__` and `Network.step` validation | `tests/test_network_validation_edges.py`, `tests/test_network_external_input.py` | `pytest -q` |

## Notes

Safety constraints are enforced at API boundaries to prevent invalid data from
propagating into dynamics or memory updates. The test suite provides regression
coverage for these boundaries under deterministic seeds.

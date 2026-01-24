# Component Audit Table

| Component ID | SPEC section | Equations / definitions referenced | Implementation paths | Tests | Status | Notes |
|---|---|---|---|---|---|---|
| P0-1 | P0-1 | Membrane equation; adaptation equation; reset rule; exp clamp note | src/bnsyn/neuron/adex.py; src/bnsyn/sim/network.py | tests/test_adex_smoke.py; tests/validation/test_adex_validation.py | VERIFIED | No mismatches detected. |
| P0-2 | P0-2 | Synaptic current equation; Mg block equation; exponential decay update | src/bnsyn/synapse/conductance.py; src/bnsyn/sim/network.py | tests/test_synapse_smoke.py; tests/validation/test_synapse_validation.py | VERIFIED | No mismatches detected. |
| P0-3 | P0-3 | Eligibility trace equation; weight update equation; coincidence simplification; bounds | src/bnsyn/plasticity/three_factor.py | tests/test_plasticity_smoke.py; tests/validation/test_plasticity_validation.py | VERIFIED | No mismatches detected. |
| P0-4 | P0-4 | Branching ratio; gain homeostasis; clip bounds | src/bnsyn/criticality/branching.py | tests/test_criticality_smoke.py; tests/validation/test_criticality_validation.py | VERIFIED | No mismatches detected. |
| P1-5 | P1-5 | Geometric cooling; plasticity gate sigmoid | src/bnsyn/temperature/schedule.py | tests/test_temperature_smoke.py; tests/validation/test_temperature_validation.py | VERIFIED | No mismatches detected. |
| P1-6 | P1-6 | Total weight sum; fast weight dynamics; tag condition; protein threshold | src/bnsyn/consolidation/dual_weight.py | tests/test_consolidation_smoke.py; tests/validation/test_consolidation_validation.py | VERIFIED | No mismatches detected. |
| P1-7 | P1-7 | Energy regularization objective terms | src/bnsyn/energy/regularization.py | tests/test_energy_smoke.py; tests/validation/test_energy_validation.py | VERIFIED | No mismatches detected. |
| P2-8 | P2-8 | Euler/RK2/exp decay; Δt-invariance | src/bnsyn/numerics/integrators.py | tests/test_dt_invariance.py; tests/validation/test_numerics_validation.py | VERIFIED | No mismatches detected. |
| P2-9 | P2-9 | Determinism protocol; RNG injection | src/bnsyn/rng.py; src/bnsyn/sim/network.py | tests/test_determinism.py; tests/validation/test_determinism_validation.py | VERIFIED | No mismatches detected. |
| P2-10 | P2-10 | Calibration utilities (f–I fit) | src/bnsyn/calibration/fit.py | tests/test_calibration_smoke.py; tests/validation/test_calibration_validation.py | VERIFIED | No mismatches detected. |
| P2-11 | P2-11 | Reference network simulator | src/bnsyn/sim/network.py | tests/test_network_smoke.py; tests/validation/test_network_validation.py | VERIFIED | No mismatches detected. |
| P2-12 | P2-12 | Bench harness/CLI contract | src/bnsyn/cli.py | tests/test_cli_smoke.py; tests/validation/test_cli_validation.py | VERIFIED | No mismatches detected. |

## Proof logs (summary)

- `python -m pip install -e ".[dev]"` → succeeded; editable install completed.
- `python scripts/validate_bibliography.py` → OK: bibliography SSOT validated.
- `python scripts/validate_claims.py` → [claims-gate] OK: 19 claims validated; 16 normative.
- `pytest -m "not validation"` → 19 passed, 33 deselected.
- `pytest -m validation` → 33 passed, 19 deselected.
- `python scripts/audit_spec_implementation.py` → Spec implementation audit passed.

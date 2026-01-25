# Physics Dashboard

This dashboard tracks numeric invariants and regression gates derived from the deterministic
BN-Syn simulation harness and validation tests.

| Metric | Test | Code | Claim | Paper |
| --- | --- | --- | --- | --- |
| σ (branching ratio) | [`tests/validation/test_physical_invariants.py`](../tests/validation/test_physical_invariants.py) | [`src/bnsyn/metrics.py`](../src/bnsyn/metrics.py) | [CLM-0007](../claims/claims.yml) | [beggs2003neuronal](../bibliography/bnsyn.bib) |
| Entropy rate | [`tests/validation/test_physical_invariants.py`](../tests/validation/test_physical_invariants.py) | [`src/bnsyn/metrics.py`](../src/bnsyn/metrics.py) | [CLM-0019](../claims/claims.yml) | [kirkpatrick1983annealing](../bibliography/bnsyn.bib) |
| Power-law α | [`tests/validation/test_physical_invariants.py`](../tests/validation/test_physical_invariants.py) | [`src/bnsyn/metrics.py`](../src/bnsyn/metrics.py) | [CLM-0006](../claims/claims.yml) | [beggs2003neuronal](../bibliography/bnsyn.bib) |
| Plasticity energy | [`tests/validation/test_physical_invariants.py`](../tests/validation/test_physical_invariants.py) | [`src/bnsyn/metrics.py`](../src/bnsyn/metrics.py) | [CLM-0021](../claims/claims.yml) | [hopfield1982neural](../bibliography/bnsyn.bib) |
| Temperature phase | [`tests/validation/test_physical_invariants.py`](../tests/validation/test_physical_invariants.py) | [`src/bnsyn/metrics.py`](../src/bnsyn/metrics.py) | [CLM-0019](../claims/claims.yml) | [kirkpatrick1983annealing](../bibliography/bnsyn.bib) |
| dt error | [`tests/validation/test_physical_invariants.py`](../tests/validation/test_physical_invariants.py) | [`tests/validation/test_physical_invariants.py`](../tests/validation/test_physical_invariants.py) | [CLM-0022](../claims/claims.yml) | [hairer1993solving](../bibliography/bnsyn.bib) |
| Runtime | [`tests/benchmarks/test_pytest_benchmark.py`](../tests/benchmarks/test_pytest_benchmark.py) | [`tests/benchmarks/test_pytest_benchmark.py`](../tests/benchmarks/test_pytest_benchmark.py) | [CLM-0022](../claims/claims.yml) | [hairer1993solving](../bibliography/bnsyn.bib) |
| Memory | [`tests/validation/test_physical_invariants.py`](../tests/validation/test_physical_invariants.py) | [`tests/validation/test_physical_invariants.py`](../tests/validation/test_physical_invariants.py) | [CLM-0021](../claims/claims.yml) | [hopfield1982neural](../bibliography/bnsyn.bib) |

Baseline values are stored in [`benchmarks/baseline.json`](../benchmarks/baseline.json) and
validated in CI by `scripts/compare_physics_baseline.py`.

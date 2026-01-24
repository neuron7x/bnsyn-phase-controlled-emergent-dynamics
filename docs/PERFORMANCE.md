# Performance Benchmarks (Deterministic)

## Purpose

This document defines the deterministic, audit-grade performance and scalability benchmarks for
BN-Syn. The benchmarks quantify CPU cost, memory usage, dt-invariance drift, and criticality drift
under load without altering scientific logic.

## Metrics

| Metric | Units | Description |
| --- | --- | --- |
| adex_steps_per_sec | steps/sec | AdEx step throughput |
| memory_per_neuron_mb | MB/neuron | Peak RSS normalized by neuron count |
| synapse_update_cost_ms | ms/step | Conductance synapse update cost |
| plasticity_update_cost_ms | ms/step | Three-factor plasticity update cost |
| dt_invariance_drift | relative | Max relative drift (σ, rate) for dt vs dt/2 |
| criticality_sigma_drift | abs(sigma-target) | Mean σ drift from target |

## Scenarios

Fixed scenarios (seed = 42):

- N = 1k, 10k, 100k neurons
- dt = 0.1 ms, 0.05 ms

CI uses the bounded scenario: N = 1k, dt = 0.1 ms.

## Thresholds

- dt_invariance_drift ≤ 0.05
- criticality_sigma_drift ≤ 0.20
- Throughput must be non-increasing with N (within 5% tolerance)
- Per-step costs must be non-decreasing with N (within 5% tolerance)

## SPEC + Claims Mapping

| Metric | SPEC | Claim | Bibkey |
| --- | --- | --- | --- |
| adex_steps_per_sec | P0-1 | CLM-0002 | brette2005adaptive |
| memory_per_neuron_mb | P2-11 | CLM-0025 | izhikevich2003simple |
| synapse_update_cost_ms | P0-2 | CLM-0003 | jahr1990voltage |
| plasticity_update_cost_ms | P0-3 | CLM-0004 | fremaux2016neuromodulated |
| dt_invariance_drift | P2-8 | CLM-0022 | hairer1993solving |
| criticality_sigma_drift | P0-4 | CLM-0006 | beggs2003neuronal |

## Running Benchmarks

Full suite (offline):
```bash
python scripts/run_benchmarks.py --suite full
```

CI suite (bounded):
```bash
python scripts/run_benchmarks.py --suite ci
```

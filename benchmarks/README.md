# BN-Syn Benchmarks

Performance benchmarking framework for BN-Syn scalability analysis.

## Overview

This directory contains tools for measuring BN-Syn performance with deterministic, reproducible parameter sweeps. Benchmarks measure:

- **Wall time** (total and per-step)
- **Memory usage** (peak RSS)
- **Throughput** (neuron-steps/sec)

## Quick Start

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run quick benchmark (local development):
```bash
python benchmarks/run_benchmarks.py --scenario quick --repeats 3 --out results/bench.csv
```

Generate report:
```bash
python benchmarks/report.py --input results/bench.csv --output docs/benchmarks/README.md
```

Or use Makefile targets:
```bash
make bench              # quick local benchmark
make bench-sweep        # full parameter sweep (slow)
make bench-report       # regenerate report from latest results
```

## Scenario Sets

- **ci_smoke**: Minimal scenario for CI validation (N=50, steps=100)
- **quick**: Small set for local development (N=100-500, steps=500-1000)
- **n_sweep**: Network size scalability (N=100 to 10,000)
- **steps_sweep**: Simulation length scalability (steps=500 to 5,000)
- **conn_sweep**: Connection density scalability (p_conn=0.01 to 0.2)
- **dt_sweep**: Timestep sweep for dt-invariance validation
- **full**: All sweeps combined (slow, for comprehensive analysis)

## Design

### Isolation

Each benchmark run executes in a fresh subprocess to avoid cross-run contamination. This ensures:
- Clean memory state
- Independent RNG seeding
- No accumulation of numerical drift

### Determinism

All benchmarks use:
- Fixed seeds (default: 42)
- Pinned `numpy.random.Generator` via `bnsyn.rng.seed_all()`
- Exact parameter serialization

### Metrics

**Timing:**
- `wall_time_sec`: Total elapsed time
- `per_step_ms`: Time per simulation step

**Memory:**
- `peak_rss_mb`: Process resident set size peak (MB)

**Throughput:**
- `neuron_steps_per_sec`: (N_neurons × steps) / wall_time

**Aggregation:**
- Each scenario runs with warmup + N repeats
- Results report: mean, p50, p95, std

### Output Format

**CSV/JSON:**
- One row per scenario (aggregated across repeats)
- Includes: git SHA, Python version, timestamp, all parameters, all metrics

**Markdown Report:**
- Summary table (all scenarios)
- Detailed metrics per scenario
- Auto-generated from CSV/JSON

## Reproducibility

See [`docs/benchmarks/PROTOCOL.md`](../docs/benchmarks/PROTOCOL.md) for:
- Exact environment requirements
- Interpretation guidelines
- Known variability sources

See [`docs/benchmarks/SCHEMA.md`](../docs/benchmarks/SCHEMA.md) for:
- CSV/JSON schema definitions
- Field descriptions
- Units and conventions

## Files

- `run_benchmarks.py`: CLI harness
- `scenarios.py`: Parameter sweep definitions
- `metrics.py`: Metrics collection utilities
- `report.py`: Report generator
- `README.md`: This file

## CI Integration

Optional workflow (`.github/workflows/benchmarks.yml`) runs `ci_smoke` scenario on schedule (weekly) and uploads artifacts. Not merge-blocking by default.

## Limitations

- **Runner variability**: GitHub Actions runners have variable CPU performance and throttling
- **OS noise**: Shared systems introduce timing variance
- **No GPU**: All benchmarks are CPU-only
- **Memory overhead**: psutil adds ~5-10MB baseline overhead

Expect ±5-10% timing variance on shared runners. For precise comparisons, use dedicated hardware.

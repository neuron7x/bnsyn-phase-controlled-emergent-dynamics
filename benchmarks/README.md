# BN-Syn Benchmarks

Performance benchmarking framework for deterministic, reproducible BN-Syn analysis.

## Overview

This directory contains tools for measuring BN-Syn performance with deterministic, reproducible
parameter sweeps. Benchmarks measure:

- **Wall time** (total and per-step)
- **Memory usage** (peak RSS)
- **Throughput** (neuron-steps/sec)

## Quick Start

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run deterministic microbench (CI-safe):
```bash
python scripts/run_benchmarks.py --suite micro
```

Run full deterministic suite (local):
```bash
python scripts/run_benchmarks.py --suite full --json-out benchmarks/results/bench_full.json
```

Or use Makefile targets:
```bash
make bench-micro         # CI-safe microbench
make bench               # full deterministic suite
```

JSON output is written to `benchmarks/results/` (gitignored). A compact, human-readable
summary table is printed to stdout after each run.

## Scenario Sets (deterministic runner)

- **micro**: SCN-001 smoke-bench (CI-safe)
- **full**: SCN-001 smoke-bench, SCN-002 core-step, SCN-003 scale sweep

Legacy scenario sets remain available for the CSV/JSON sweep runner:
`ci_smoke`, `quick`, `n_sweep`, `steps_sweep`, `conn_sweep`, `dt_sweep`.

## Design

### Isolation

The deterministic runner executes in-process with warmup + median measurement. The legacy
sweep runner executes each run in a fresh subprocess to avoid cross-run contamination.

### Determinism

All benchmarks use:
- Fixed seeds
- Pinned `numpy.random.Generator` via `bnsyn.rng.seed_all()`
- Exact parameter serialization
- Single-threaded BLAS via `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`

### Metrics

**Timing:**
- `wall_time_sec`: Total elapsed time
- `per_step_ms`: Time per simulation step

**Memory:**
- `peak_rss_mb`: Process resident set size peak (MB)

**Throughput:**
- `neuron_steps_per_sec`: (N_neurons × steps) / wall_time

**Aggregation (deterministic runner):**
- Warmup run + 3 measured runs
- Report median metrics per scenario

**Aggregation (legacy sweep runner):**
- Warmup + N repeats
- Results report: mean, p50, p95, std

### Output Format

**JSON (deterministic runner):**
- `metadata`, `suite`, `results`
- Each result contains `scenario`, `runs`, and `summary`

**CSV/JSON (legacy sweep runner):**
- One row per scenario (aggregated across repeats)
- Includes: git SHA, Python version, timestamp, all parameters, all metrics

## Reproducibility

See [`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md) for deterministic benchmark usage and
interpretation guidelines. For legacy sweep details, see
[`docs/benchmarks/PROTOCOL.md`](../docs/benchmarks/PROTOCOL.md) and
[`docs/benchmarks/SCHEMA.md`](../docs/benchmarks/SCHEMA.md).

## Files

- `runner.py`: Deterministic benchmark runner
- `reporting.py`: Summary table renderer
- `scenarios.py`: Deterministic and legacy scenario definitions
- `run_benchmarks.py`: Legacy CSV/JSON sweep runner
- `metrics.py`: Legacy metrics collection utilities
- `report.py`: Legacy report generator
- `README.md`: This file

## CI Integration

Optional workflow (`.github/workflows/benchmarks.yml`) runs `ci_smoke` scenario on schedule (weekly) and uploads artifacts. Not merge-blocking by default.

## Limitations

- **Runner variability**: GitHub Actions runners have variable CPU performance and throttling
- **OS noise**: Shared systems introduce timing variance
- **No GPU**: All benchmarks are CPU-only
- **Memory overhead**: psutil adds ~5-10MB baseline overhead

Expect ±5-10% timing variance on shared runners. For precise comparisons, use dedicated hardware.

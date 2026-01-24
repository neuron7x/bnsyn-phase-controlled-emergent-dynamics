# Performance Benchmarks (Deterministic)

## Purpose

This document defines the deterministic performance benchmark layer for BN-Syn. The benchmarks:

- Measure **runtime** and **peak RSS memory** with fixed inputs.
- Provide **reproducible, deterministic** runs for tracking performance regressions.
- Separate **benchmarks** from **tests** so performance tracking never blocks correctness checks.

## Non-goals

- These benchmarks do **not** validate scientific correctness (see tests/validation instead).
- They do **not** replace the existing benchmark/validation suite under `benchmarks/` (which targets
  physics, stability, and regression metrics).
- They do **not** attempt to measure GPU performance.

## Quick Commands

Microbench (fast, deterministic; designed for CI):

```bash
make bench-micro
```

Full local suite (includes scale sweep):

```bash
make bench
```

Direct script usage:

```bash
python scripts/run_benchmarks.py --suite micro
python scripts/run_benchmarks.py --suite full --json-out benchmarks/results/full.json
```

## Scenario Definitions

Each scenario is fully deterministic: fixed seed, fixed step count, and fixed network parameters.

| Scenario ID | Name | Purpose | Seed | Steps | dt-ms | Size |
|-------------|------|---------|------|-------|-------|------|
| SCN-001 | smoke-bench | CI-safe smoke benchmark | 202401 | 50 | 0.5 | N=64 |
| SCN-002 | core-step | Representative local run | 202402 | 200 | 0.5 | N=256 |
| SCN-003 | scale-sweep | Local-only scalability sweep | 202403 | 150 | 0.5 | N=128/256/512 |

## What Each Scenario Measures

- **SCN-001 smoke-bench:** fast runtime + memory regression signal (bounded < ~60s).
- **SCN-002 core-step:** stable medium-size performance signature for local profiling.
- **SCN-003 scale-sweep:** scaling behavior across representative network sizes.

## Interpretation Guidelines

Benchmarks use **1 warmup + 3 measured runs** and report the **median** for:

- `wall_time_sec`: elapsed wall clock time per scenario.
- `peak_rss_mb`: max resident set size (RSS) in MB on Linux.
- `per_step_ms`: average milliseconds per simulation step.

Expected variability on the same machine should be within **±5%** for wall time and per-step
metrics. Peak RSS should be stable within **±2%**. Larger drift suggests a real regression or a
change in runtime environment (CPU scaling, background load, Python version).

## Determinism Rules

The benchmark runner enforces:

- Fixed seeds via `bnsyn.rng.seed_all`.
- Fixed network parameters and step counts.
- Thread pinning via `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.

Avoid changing seeds, step counts, or size parameters unless you introduce a **new** scenario
identifier. Keep existing scenario IDs stable to preserve long-term comparability.

## Extending Benchmarks Safely

When adding a new scenario:

1. Use a new `SCN-XXX` identifier.
2. Fix seed, steps, dt-ms, and size parameters explicitly.
3. Keep runtime bounded and deterministic.
4. Update this document and ensure the scenario is added to `benchmarks/scenarios.py`.

## Outputs

Runs produce:

- **Human-readable table** printed to stdout.
- **JSON report** saved under `benchmarks/results/*.json` (gitignored).

The JSON includes system metadata (Python, platform, CPU info, git SHA) alongside per-scenario
metrics for reproducibility and audit trails.

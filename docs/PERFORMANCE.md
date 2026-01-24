# BN-Syn Performance Benchmarks

## Purpose (and Non-goals)

**Purpose:** Provide deterministic, reproducible performance baselines for BN-Syn runtime and memory across a fixed set of scenarios. Benchmarks are separated from correctness tests and are intended for **trend tracking**, not absolute hardware comparisons.

**Non-goals:**
- These benchmarks do **not** validate scientific correctness or numerical outputs.
- They do **not** attempt to normalize results across different CPU architectures.
- They are **not** intended to be exhaustive stress tests.

## Determinism Protocol

The benchmark runner enforces deterministic behavior by:
- Fixed seeds per scenario.
- Fixed steps and dt per scenario.
- Forcing single-threaded BLAS via `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.
- Running a warmup (1) and reporting the **median** of 3 measured runs.

RSS memory is measured via `resource.getrusage().ru_maxrss` on Linux. On non-Linux platforms, RSS is reported as `null` (not available in a comparable way).

## Running Benchmarks

### Microbench (CI-safe)

```bash
make bench-micro
```

### Full Suite (local)

```bash
make bench
```

### Direct CLI

```bash
python scripts/run_benchmarks.py --suite micro
python scripts/run_benchmarks.py --suite full --json-out benchmarks/results/bench_full.json
```

JSON results are written under `benchmarks/results/*.json` (gitignored).

## Scenario Definitions

| ID | Name | Seed | Steps | dt (ms) | N | p_conn | frac_inhib | Purpose |
|----|------|------|-------|---------|---|--------|------------|---------|
| SCN-001 | smoke-bench | 4242 | 120 | 0.1 | 64 | 0.05 | 0.2 | Tiny deterministic run for CI microbench |
| SCN-002 | core-step | 4242 | 800 | 0.1 | 512 | 0.05 | 0.2 | Representative stepping workload |
| SCN-003 | scale-sweep-* | 4242 | 600 | 0.1 | 256/512/1024/2048 | 0.05 | 0.2 | Network size sweep |

## Interpreting Results

- **Same machine comparisons:** Expect median wall time to remain within **±5%** for small changes. Larger deviations indicate regression or environmental noise.
- **Different machines:** Compare **relative trends** (e.g., scaling curve) rather than absolute numbers.
- **RSS:** On Linux, `ru_maxrss` is reported in MB. This value is process-wide peak RSS, so it may include allocations retained by previous runs.

## Extending Benchmarks Safely

When adding scenarios:
1. Use **fixed seeds** and explicit parameters.
2. Keep warmup+median logic intact.
3. Do not change BN-Syn core logic or RNG semantics.
4. Keep microbench fast (<30s on GH runner).
5. Document any new scenario in this file with exact parameters.

Benchmarks should remain deterministic and must never gate correctness tests.

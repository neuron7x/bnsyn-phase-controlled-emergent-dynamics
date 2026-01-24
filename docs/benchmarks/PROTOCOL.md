# BN-Syn Benchmark Protocol

Reproducibility protocol for BN-Syn performance benchmarks.

## Determinism Guarantees

All benchmarks are deterministic given fixed environment and seeds:

1. **RNG seeding**: All randomness flows through `bnsyn.rng.seed_all(seed)` which seeds:
   - `numpy.random.Generator` (primary)
   - Python `random` module (fallback)
   - `PYTHONHASHSEED` environment variable

2. **Parameter serialization**: All scenario parameters are serialized to JSON/CSV with exact types

3. **Subprocess isolation**: Each run executes in a fresh subprocess with clean state

4. **No hidden state**: BN-Syn core has no module-level globals that accumulate state

## Environment Requirements

### Pinned Dependencies

Install exact versions:
```bash
pip install -e ".[dev]"
```

Core dependencies (from `pyproject.toml`):
- `numpy>=1.26`
- `scipy>=1.10`
- `psutil>=5.9` (benchmark-specific)

### Python Version

Tested on Python 3.11+. Results may vary on older Python versions due to NumPy implementation changes.

### Operating System

Benchmarks are OS-agnostic but timing will vary:
- **Linux**: Most consistent (recommended for CI)
- **macOS**: Generally fast but subject to CPU throttling
- **Windows**: Higher OS noise

## Running Benchmarks

### Minimal Example (CI Smoke Test)

```bash
python benchmarks/run_benchmarks.py \
  --scenario ci_smoke \
  --repeats 3 \
  --out results/ci_smoke.csv \
  --json results/ci_smoke.json
```

Expected runtime: <10 seconds

### Quick Local Benchmark

```bash
python benchmarks/run_benchmarks.py \
  --scenario quick \
  --repeats 3 \
  --out results/quick.csv
```

Expected runtime: 1-2 minutes

### Full Parameter Sweep

```bash
python benchmarks/run_benchmarks.py \
  --scenario full \
  --repeats 5 \
  --out results/full.csv
```

Expected runtime: 10-30 minutes (depends on hardware)

### Generate Report

```bash
python benchmarks/report.py \
  --input results/quick.csv \
  --output docs/benchmarks/README.md
```

## Interpreting Results

### Scalability Parameters

| Parameter | Complexity Driver | Expected Scaling |
|-----------|------------------|------------------|
| `N_neurons` | Network size | O(N) time, O(N) memory for sparse connectivity |
| `steps` | Simulation length | O(steps) time, O(1) memory |
| `p_conn` | Connection density | O(N² × p_conn) memory, O(N × fan_in) time per step |
| `dt_ms` | Timestep | O(1/dt) time for fixed real-time duration |

### Metric Definitions

**wall_time_sec:**
- Total elapsed wall-clock time (includes Python overhead)
- Use for absolute performance assessment

**per_step_ms:**
- Time per simulation step (wall_time / steps × 1000)
- Use for comparing step efficiency across scenarios

**peak_rss_mb:**
- Peak resident set size (process memory footprint)
- Includes Python interpreter, NumPy arrays, overhead
- Use for memory scaling analysis

**neuron_steps_per_sec:**
- Throughput metric: (N_neurons × steps) / wall_time
- Use for comparing hardware efficiency
- Higher is better

**spike_count_total:**
- Total spikes across all runs (diagnostic)
- Should be consistent across repeats for same scenario (determinism check)

### Expected Variance

**Shared/CI Runners:**
- Timing: ±5-10% (CPU throttling, OS noise)
- Memory: ±2-5% (Python GC non-determinism)

**Dedicated Hardware:**
- Timing: ±1-3% (cache effects, OS scheduling)
- Memory: ±1% (Python GC non-determinism)

### Regression Detection

**Significant regression:**
- >15% increase in wall_time (same hardware)
- >10% increase in peak_rss (same hardware)

**Investigate if:**
- Throughput drops >10%
- Spike count changes (determinism violation)

**Likely false positive:**
- <5% changes on shared runners
- Single-run outliers (check p95 vs mean)

## Baseline Establishment

To establish a baseline for regression tracking:

1. Run full sweep with 5+ repeats on dedicated hardware
2. Record: hardware spec, OS, Python version, git SHA
3. Store results as `baselines/{sha}.csv`
4. Use p50 values for comparison (robust to outliers)

## Limitations

### Known Variability Sources

1. **CPU throttling**: Thermal or power management can reduce clock speed mid-run
2. **OS scheduler**: Non-real-time scheduling introduces jitter
3. **Python GC**: Non-deterministic garbage collection timing
4. **Shared caches**: Other processes evict cache lines
5. **Turbo boost**: CPU frequency varies with thermal headroom

### Not Measured

- **Energy consumption**: Requires hardware counters
- **GPU performance**: BN-Syn is CPU-only
- **I/O time**: Benchmarks are compute-bound
- **Network latency**: Single-process only

### Excluded from Scope

- **Model accuracy**: See validation tests in `tests/validation/`
- **Numerical stability**: See dt-invariance tests
- **Determinism**: See `tests/test_determinism.py`

## CI Integration

The optional `benchmarks.yml` workflow:
- Runs `ci_smoke` scenario (N=50, steps=100)
- Uploads CSV/JSON artifacts
- Does NOT block PR merges by default
- Scheduled weekly for trend tracking

To make merge-blocking (not recommended without baselines):
- Add workflow to required checks in branch protection
- Implement automated baseline comparison with thresholds

## Changelog

- **2026-01-24**: Initial protocol (git SHA: TBD)

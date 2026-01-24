# BN-Syn Benchmark Results Schema

Schema definition for CSV/JSON benchmark results.

## Output Format

Benchmark results are written in two formats:
- **CSV**: One row per scenario (with headers)
- **JSON**: Array of objects (one per scenario)

Both formats contain identical fields with identical semantics.

## Field Definitions

### Metadata Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `scenario` | string | Unique scenario name | `"n_sweep_1000"` |
| `git_sha` | string | Git commit SHA (full 40 chars) | `"a1b2c3d4..."` |
| `python_version` | string | Python version (major.minor.micro) | `"3.11.5"` |
| `timestamp` | string | UTC timestamp (ISO 8601) | `"2026-01-24T12:34:56.789Z"` |
| `description` | string | Human-readable scenario description | `"Network size sweep: N=1000"` |
| `repeats` | integer | Number of successful runs | `3` |

### Scenario Parameters

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `seed` | integer | - | RNG seed for deterministic runs |
| `dt_ms` | float | milliseconds | Integration timestep |
| `steps` | integer | - | Number of simulation steps |
| `N_neurons` | integer | - | Total neurons (excitatory + inhibitory) |
| `p_conn` | float | - | Connection probability (0 to 1) |
| `frac_inhib` | float | - | Fraction of inhibitory neurons (0 to 1) |

### Timing Metrics

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `wall_time_sec_mean` | float | seconds | Mean wall-clock time across repeats |
| `wall_time_sec_p50` | float | seconds | Median (50th percentile) wall-clock time |
| `wall_time_sec_p95` | float | seconds | 95th percentile wall-clock time |
| `wall_time_sec_std` | float | seconds | Standard deviation of wall-clock time |
| `per_step_ms_mean` | float | milliseconds | Mean time per simulation step |
| `per_step_ms_p50` | float | milliseconds | Median time per simulation step |
| `per_step_ms_p95` | float | milliseconds | 95th percentile time per step |
| `per_step_ms_std` | float | milliseconds | Standard deviation of time per step |

**Definition:**
```
per_step_ms = (wall_time_sec / steps) * 1000
```

### Memory Metrics

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `peak_rss_mb_mean` | float | megabytes | Mean peak resident set size |
| `peak_rss_mb_p50` | float | megabytes | Median peak RSS |
| `peak_rss_mb_p95` | float | megabytes | 95th percentile peak RSS |
| `peak_rss_mb_std` | float | megabytes | Standard deviation of peak RSS |

**Definition:**
- Measured via `psutil.Process().memory_info().rss`
- Peak during run (max of start and end RSS)
- Includes Python interpreter, libraries, and all allocations

### Throughput Metrics

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `neuron_steps_per_sec_mean` | float | neuron-steps/sec | Mean throughput |
| `neuron_steps_per_sec_p50` | float | neuron-steps/sec | Median throughput |
| `neuron_steps_per_sec_p95` | float | neuron-steps/sec | 95th percentile throughput |
| `neuron_steps_per_sec_std` | float | neuron-steps/sec | Standard deviation of throughput |

**Definition:**
```
neuron_steps_per_sec = (N_neurons * steps) / wall_time_sec
```

Interpretation:
- Measures computational throughput (higher is better)
- Accounts for both network size and simulation length
- Use for cross-hardware comparisons

### Activity Metrics

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `spike_count_total` | integer | spikes | Total spikes summed across all repeats |

**Definition:**
- Sum of all spikes detected across all neurons, all steps, all repeats
- Diagnostic metric (should be deterministic for same seed)

## Example Row (CSV)

```csv
scenario,git_sha,python_version,timestamp,name,seed,dt_ms,steps,N_neurons,p_conn,frac_inhib,description,repeats,wall_time_sec_mean,wall_time_sec_p50,wall_time_sec_p95,wall_time_sec_std,peak_rss_mb_mean,peak_rss_mb_p50,peak_rss_mb_p95,peak_rss_mb_std,per_step_ms_mean,per_step_ms_p50,per_step_ms_p95,per_step_ms_std,neuron_steps_per_sec_mean,neuron_steps_per_sec_p50,neuron_steps_per_sec_p95,neuron_steps_per_sec_std,spike_count_total
n_sweep_1000,a1b2c3d4e5f6...,3.11.5,2026-01-24T12:34:56.789Z,n_sweep_1000,42,0.1,500,1000,0.05,0.2,"Network size sweep: N=1000",3,2.451,2.450,2.475,0.012,125.3,125.1,126.2,0.5,4.902,4.900,4.950,0.024,204082,204164,202020,1020,15234
```

## Example Object (JSON)

```json
{
  "scenario": "n_sweep_1000",
  "git_sha": "a1b2c3d4e5f6...",
  "python_version": "3.11.5",
  "timestamp": "2026-01-24T12:34:56.789Z",
  "name": "n_sweep_1000",
  "seed": 42,
  "dt_ms": 0.1,
  "steps": 500,
  "N_neurons": 1000,
  "p_conn": 0.05,
  "frac_inhib": 0.2,
  "description": "Network size sweep: N=1000",
  "repeats": 3,
  "wall_time_sec_mean": 2.451,
  "wall_time_sec_p50": 2.450,
  "wall_time_sec_p95": 2.475,
  "wall_time_sec_std": 0.012,
  "peak_rss_mb_mean": 125.3,
  "peak_rss_mb_p50": 125.1,
  "peak_rss_mb_p95": 126.2,
  "peak_rss_mb_std": 0.5,
  "per_step_ms_mean": 4.902,
  "per_step_ms_p50": 4.900,
  "per_step_ms_p95": 4.950,
  "per_step_ms_std": 0.024,
  "neuron_steps_per_sec_mean": 204082,
  "neuron_steps_per_sec_p50": 204164,
  "neuron_steps_per_sec_p95": 202020,
  "neuron_steps_per_sec_std": 1020,
  "spike_count_total": 15234
}
```

## Units Reference

| Quantity | Unit | Symbol |
|----------|------|--------|
| Time | seconds | s |
| Time | milliseconds | ms |
| Memory | megabytes | MB |
| Timestep | milliseconds | ms |
| Throughput | neuron-steps per second | neuron-steps/sec |

**Conversions:**
- 1 second = 1000 milliseconds
- 1 megabyte = 1024 × 1024 bytes
- 1 neuron-step = 1 neuron simulated for 1 timestep

## Aggregation Statistics

All metrics report 4 aggregation statistics:
- **mean**: Arithmetic mean across repeats
- **p50**: Median (50th percentile) - robust to outliers
- **p95**: 95th percentile - captures worst-case behavior
- **std**: Standard deviation - measures variance

Use **p50** for typical performance, **p95** for worst-case, **std** for stability assessment.

## Changelog

- **2026-01-24**: Initial schema (git SHA: cc3b5f0c8d75c398a488d70390e9917cc720ba21)

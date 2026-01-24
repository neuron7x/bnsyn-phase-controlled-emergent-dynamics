#!/usr/bin/env python3
"""Benchmark harness for BN-Syn scalability testing.

Runs benchmark scenarios with subprocess isolation for clean measurements.
Outputs machine-readable CSV/JSON and human-readable console logs.

Usage:
    python benchmarks/run_benchmarks.py --scenario quick --repeats 3 --out results/bench.csv
    python benchmarks/run_benchmarks.py --scenario full --repeats 5 --out results/bench_full.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.scenarios import get_scenarios


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def run_single_benchmark(scenario_dict: dict[str, Any]) -> dict[str, Any]:
    """Run a single benchmark scenario in-process (called by subprocess)."""
    import gc
    import time

    import numpy as np

    from bnsyn.config import AdExParams, CriticalityParams, SynapseParams
    from bnsyn.rng import seed_all
    from bnsyn.sim.network import Network, NetworkParams

    # Extract scenario params
    seed = scenario_dict["seed"]
    dt_ms = scenario_dict["dt_ms"]
    steps = scenario_dict["steps"]
    N = scenario_dict["N_neurons"]
    p_conn = scenario_dict["p_conn"]
    frac_inhib = scenario_dict["frac_inhib"]

    # Seed all RNGs
    pack = seed_all(seed)
    rng = pack.np_rng

    # Force GC before benchmark
    gc.collect()

    # Setup network
    nparams = NetworkParams(N=N, p_conn=p_conn, frac_inhib=frac_inhib)
    net = Network(
        nparams,
        AdExParams(),
        SynapseParams(),
        CriticalityParams(),
        dt_ms=dt_ms,
        rng=rng,
    )

    # Track metrics
    import psutil

    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss / (1024 * 1024)  # MB
    start_time = time.perf_counter()

    spike_count = 0
    # Run simulation
    for _ in range(steps):
        metrics = net.step()
        spike_count += int(metrics["A_t1"])

    # Finalize metrics
    wall_time = time.perf_counter() - start_time
    end_rss = process.memory_info().rss / (1024 * 1024)  # MB
    peak_rss = max(end_rss, start_rss)

    per_step_ms = (wall_time / steps) * 1000.0
    neuron_steps = N * steps
    neuron_steps_per_sec = neuron_steps / wall_time

    return {
        "wall_time_sec": wall_time,
        "peak_rss_mb": peak_rss,
        "per_step_ms": per_step_ms,
        "neuron_steps_per_sec": neuron_steps_per_sec,
        "spike_count": spike_count,
    }


def run_scenario_subprocess(
    scenario_dict: dict[str, Any], repeat_idx: int
) -> dict[str, Any] | None:
    """Run a benchmark scenario in a subprocess for isolation."""
    # Prepare subprocess script
    script = f"""
import sys
import json
sys.path.insert(0, {repr(str(Path.cwd()))})

from benchmarks.run_benchmarks import run_single_benchmark

scenario = {repr(scenario_dict)}
result = run_single_benchmark(scenario)
print(json.dumps(result))
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout per run
            check=True,
        )
        metrics = json.loads(result.stdout.strip())
        return metrics
    except subprocess.TimeoutExpired:
        print(f"  WARNING: Repeat {repeat_idx} timed out", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  WARNING: Repeat {repeat_idx} failed: {e}", file=sys.stderr)
        return None


def aggregate_metrics(runs: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate metrics across multiple runs."""
    import numpy as np

    if not runs:
        return {}

    keys = ["wall_time_sec", "peak_rss_mb", "per_step_ms", "neuron_steps_per_sec"]
    agg = {}
    for key in keys:
        values = [r[key] for r in runs]
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_p50"] = float(np.percentile(values, 50))
        agg[f"{key}_p95"] = float(np.percentile(values, 95))
        agg[f"{key}_std"] = float(np.std(values))

    # Spike count is summed
    agg["spike_count_total"] = sum(r["spike_count"] for r in runs)

    return agg


def run_benchmarks(
    scenario_set: str,
    repeats: int,
    output_csv: str | None,
    output_json: str | None,
    warmup: bool = True,
) -> None:
    """Run benchmark scenarios and write results."""
    scenarios = get_scenarios(scenario_set)
    git_sha = get_git_sha()
    python_ver = get_python_version()
    timestamp = datetime.now().astimezone().isoformat()

    print(f"Running {len(scenarios)} scenarios with {repeats} repeats each")
    print(f"Git SHA: {git_sha}")
    print(f"Python: {python_ver}")
    print(f"Timestamp: {timestamp}")
    print()

    all_results = []

    for idx, scenario in enumerate(scenarios):
        print(
            f"[{idx+1}/{len(scenarios)}] {scenario.name}: "
            f"N={scenario.N_neurons}, steps={scenario.steps}, dt={scenario.dt_ms}ms"
        )

        scenario_dict = scenario.to_dict()

        # Warmup run (not recorded)
        if warmup:
            print("  Warmup...", end="", flush=True)
            _ = run_scenario_subprocess(scenario_dict, 0)
            print(" done")

        # Actual runs
        runs = []
        for r in range(repeats):
            print(f"  Repeat {r+1}/{repeats}...", end="", flush=True)
            result = run_scenario_subprocess(scenario_dict, r + 1)
            if result is not None:
                runs.append(result)
                print(
                    f" {result['wall_time_sec']:.2f}s, "
                    f"{result['peak_rss_mb']:.1f}MB, "
                    f"{result['neuron_steps_per_sec']:.0f} neuron-steps/sec"
                )
            else:
                print(" FAILED")

        if not runs:
            print("  ERROR: All runs failed, skipping scenario")
            continue

        # Aggregate
        agg = aggregate_metrics(runs)

        # Build result row
        result_row = {
            "scenario": scenario.name,
            "git_sha": git_sha,
            "python_version": python_ver,
            "timestamp": timestamp,
            **scenario_dict,
            "repeats": len(runs),
            **agg,
        }
        all_results.append(result_row)

        print(
            f"  Summary: {agg['wall_time_sec_mean']:.2f}s (±{agg['wall_time_sec_std']:.2f}), "
            f"{agg['peak_rss_mb_mean']:.1f}MB, "
            f"{agg['neuron_steps_per_sec_mean']:.0f} neuron-steps/sec"
        )
        print()

    # Write CSV
    if output_csv and all_results:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(all_results[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Wrote CSV: {output_csv}")

    # Write JSON
    if output_json and all_results:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Wrote JSON: {output_json}")

    print(f"\nCompleted {len(all_results)} scenarios")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BN-Syn benchmarks")
    parser.add_argument(
        "--scenario",
        default="quick",
        help="Scenario set to run (ci_smoke, quick, n_sweep, steps_sweep, conn_sweep, dt_sweep, full)",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Number of repeats per scenario"
    )
    parser.add_argument("--out", help="Output CSV file path")
    parser.add_argument("--json", help="Output JSON file path")
    parser.add_argument(
        "--no-warmup", action="store_true", help="Skip warmup runs"
    )

    args = parser.parse_args()

    run_benchmarks(
        scenario_set=args.scenario,
        repeats=args.repeats,
        output_csv=args.out,
        output_json=args.json,
        warmup=not args.no_warmup,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

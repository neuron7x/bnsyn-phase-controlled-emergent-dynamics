#!/usr/bin/env python3
"""Benchmark harness for BN-Syn benchmarking and validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.metrics import metrics_to_dict, run_benchmark
from benchmarks.scenarios import get_scenarios
from benchmarks.scenarios.base import BenchmarkScenario


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


def run_scenario_subprocess(scenario_dict: dict[str, Any]) -> dict[str, Any] | None:
    """Run a benchmark scenario in a subprocess for isolation."""
    script = f"""
import sys
import json
sys.path.insert(0, {repr(str(Path.cwd()))})

from benchmarks.metrics import metrics_to_dict, run_benchmark
from benchmarks.scenarios.base import BenchmarkScenario

scenario = BenchmarkScenario(**{repr(scenario_dict)})
result = run_benchmark(scenario)
print(json.dumps(metrics_to_dict(result)))
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=900,
            check=True,
        )
        return json.loads(result.stdout.strip())
    except subprocess.TimeoutExpired:
        print("  WARNING: scenario timed out", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  WARNING: scenario failed: {exc}", file=sys.stderr)
        return None


def aggregate_metrics(runs: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate metrics across multiple runs using mean."""
    import numpy as np

    if not runs:
        return {}

    keys = list(runs[0].keys())
    aggregated: dict[str, float] = {}
    for key in keys:
        values = [float(run[key]) for run in runs]
        aggregated[key] = float(np.mean(values))
    return aggregated


def run_benchmarks(
    scenario_set: str,
    repeats: int,
    output_json: str | None,
) -> list[dict[str, Any]]:
    """Run benchmark scenarios and return results."""
    scenarios = get_scenarios(scenario_set)
    git_sha = get_git_sha()
    python_ver = get_python_version()
    timestamp = datetime.now().astimezone().isoformat()

    print(f"Running {len(scenarios)} scenarios with {repeats} repeats each")
    print(f"Git SHA: {git_sha}")
    print(f"Python: {python_ver}")
    print(f"Timestamp: {timestamp}\n")

    all_results: list[dict[str, Any]] = []

    for idx, scenario in enumerate(scenarios):
        print(
            f"[{idx + 1}/{len(scenarios)}] {scenario.name}: "
            f"N={scenario.N_neurons}, steps={scenario.steps}, dt={scenario.dt_ms}ms"
        )
        scenario_dict = scenario.to_dict()

        runs = []
        for repeat in range(repeats):
            print(f"  Repeat {repeat + 1}/{repeats}...", end="", flush=True)
            metrics = run_scenario_subprocess(scenario_dict)
            if metrics is None:
                print(" FAILED")
                continue
            runs.append(metrics)
            print(
                f" wall_time={metrics['performance_wall_time_sec']:.3f}s, "
                f"rss={metrics['performance_peak_rss_mb']:.1f}MB, "
                f"sigma={metrics['physics_sigma_mean']:.3f}"
            )

        if not runs:
            print("  ERROR: All runs failed, skipping scenario")
            continue

        aggregated = aggregate_metrics(runs)
        all_results.append(
            {
                "scenario": scenario.name,
                "git_sha": git_sha,
                "python_version": python_ver,
                "timestamp": timestamp,
                **scenario_dict,
                "repeats": len(runs),
                **aggregated,
            }
        )
        print(f"  Summary: wall_time={aggregated['performance_wall_time_sec']:.3f}s")
        print()

    if output_json and all_results:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Wrote JSON: {output_json}")

    return all_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BN-Syn benchmarks")
    parser.add_argument(
        "--scenario",
        default="small_network",
        help=(
            "Scenario set to run (small_network, medium_network, large_network, "
            "criticality_sweep, temperature_sweep, dt_sweep, full)"
        ),
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats per scenario")
    parser.add_argument("--json", help="Output JSON file path")

    args = parser.parse_args()

    run_benchmarks(
        scenario_set=args.scenario,
        repeats=args.repeats,
        output_json=args.json,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

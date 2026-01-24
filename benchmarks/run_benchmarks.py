#!/usr/bin/env python3
"""Benchmark harness for BN-Syn benchmarking and validation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.scenarios import get_scenarios

try:
    import torch
    from torch.utils.bottleneck import bottleneck
except Exception:  # pragma: no cover - optional GPU tooling
    torch = None
    bottleneck = None


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
import atexit
import shutil
import tempfile
import resource
import sys
import json
sys.path.insert(0, {repr(str(Path.cwd()))})

from benchmarks.metrics import metrics_to_dict, run_benchmark
from benchmarks.scenarios.base import BenchmarkScenario

tmpdir = tempfile.mkdtemp(prefix="bnsyn-bench-")
atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

limit_bytes = 1024 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

scenario = BenchmarkScenario(**{repr(scenario_dict)})
result = run_benchmark(scenario)
print(json.dumps(metrics_to_dict(result)))
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
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
    from scipy.stats import zscore

    if not runs:
        return {}

    keys = list(runs[0].keys())
    aggregated: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(run[key]) for run in runs], dtype=np.float64)
        if values.size >= 3:
            z = np.abs(zscore(values, nan_policy="omit"))
            values = values[z <= 2.0]
        if values.size == 0:
            values = np.asarray([float(run[key]) for run in runs], dtype=np.float64)
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler("bench.log", mode="w"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Running %d scenarios with %d repeats each", len(scenarios), repeats)
    logging.info("Git SHA: %s", git_sha)
    logging.info("Python: %s", python_ver)
    logging.info("Timestamp: %s", timestamp)

    device = None
    if torch is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        logging.info("Torch device: %s", device)
        os.environ["BNSYN_USE_TORCH"] = "1"
        os.environ["BNSYN_DEVICE"] = str(device)

    all_results: list[dict[str, Any]] = []

    for idx, scenario in enumerate(scenarios):
        logging.info(
            f"[{idx + 1}/{len(scenarios)}] {scenario.name}: "
            f"N={scenario.N_neurons}, steps={scenario.steps}, dt={scenario.dt_ms}ms"
        )
        scenario_dict = scenario.to_dict()

        runs = []
        for repeat in range(repeats):
            logging.info("  Repeat %d/%d...", repeat + 1, repeats)
            metrics = run_scenario_subprocess(scenario_dict)
            if metrics is None:
                logging.info("  FAILED")
                continue
            runs.append(metrics)
            logging.info(
                "  wall_time=%.3fs, rss=%.1fMB, sigma=%.3f",
                metrics["performance_wall_time_sec"],
                metrics["performance_peak_rss_mb"],
                metrics["physics_sigma_mean"],
            )

        if not runs:
            logging.info("  ERROR: All runs failed, skipping scenario")
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
        logging.info("  Summary: wall_time=%.3fs", aggregated["performance_wall_time_sec"])

    if output_json and all_results:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logging.info("Wrote JSON: %s", output_json)

    if bottleneck is not None and device is not None and device.type == "cuda":
        bottleneck()

    return all_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BN-Syn benchmarks")
    parser.add_argument(
        "--scenario",
        default="small_network",
        choices=[
            "small_network",
            "medium_network",
            "large_network",
            "criticality_sweep",
            "temperature_sweep",
            "dt_sweep",
            "full",
        ],
        help="Scenario set to run",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats per scenario")
    parser.add_argument("--json", help="Output JSON file path")

    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("repeats must be positive")

    run_benchmarks(
        scenario_set=args.scenario,
        repeats=args.repeats,
        output_json=args.json,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

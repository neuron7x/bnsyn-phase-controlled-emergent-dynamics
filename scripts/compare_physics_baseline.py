#!/usr/bin/env python3
"""Compare physics and benchmark metrics against stored baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Tolerance thresholds for metric comparisons
PHYSICS_TOLERANCE = 0.03  # 3% tolerance for deterministic physics metrics
ENVIRONMENT_TOLERANCE = 1.0  # 100% tolerance for environment-dependent metrics


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    return json.loads(path.read_text())


def _extract_runtime_ms(bench_data: dict[str, Any]) -> float:
    benchmarks = bench_data.get("benchmarks", [])
    for entry in benchmarks:
        if entry.get("name") == "test_demo_runtime":
            stats = entry.get("stats", {})
            mean_seconds = float(stats.get("mean", 0.0))
            return mean_seconds * 1000.0
    raise SystemExit("Benchmark 'test_demo_runtime' not found in bench.json")


def _compare_metric(
    name: str, baseline: float, current: float, tolerance: float = PHYSICS_TOLERANCE
) -> dict[str, float | str]:
    if baseline == 0.0:
        deviation = 0.0 if current == 0.0 else 1.0
    else:
        deviation = abs(current - baseline) / abs(baseline)
    status = "pass" if deviation <= tolerance else "fail"
    return {
        "metric": name,
        "baseline": float(baseline),
        "current": float(current),
        "deviation": float(deviation),
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare metrics against baseline.")
    parser.add_argument("--baseline", type=Path, default=Path("benchmarks/baseline.json"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/physics_metrics.json"))
    parser.add_argument("--bench", type=Path, default=Path("bench.json"))
    parser.add_argument("--output", type=Path, default=Path("diffs.json"))
    args = parser.parse_args()

    baseline_data = _load_json(args.baseline)
    if not isinstance(baseline_data, dict):
        raise SystemExit("Baseline JSON must be a mapping with physics metrics")

    metrics = _load_json(args.metrics)
    if not isinstance(metrics, dict):
        raise SystemExit("Metrics JSON must be a mapping")

    bench_data = _load_json(args.bench)
    runtime_ms = _extract_runtime_ms(bench_data)

    comparisons = []
    # Physics metrics with tight tolerance for deterministic results
    physics_keys = [
        "sigma",
        "entropy",
        "powerlaw_alpha",
        "plasticity_energy",
        "temperature",
        "dt_error",
    ]
    for key in physics_keys:
        if key not in baseline_data:
            raise SystemExit(f"Baseline missing key: {key}")
        if key not in metrics:
            raise SystemExit(f"Metrics missing key: {key}")
        comparisons.append(_compare_metric(key, float(baseline_data[key]), float(metrics[key]), tolerance=PHYSICS_TOLERANCE))

    # Environment-dependent metrics with relaxed tolerance
    # (runtime and memory vary significantly across systems)
    if "memory_mb" in baseline_data and "memory_mb" in metrics:
        comparisons.append(_compare_metric("memory_mb", float(baseline_data["memory_mb"]), float(metrics["memory_mb"]), tolerance=ENVIRONMENT_TOLERANCE))

    if "runtime_ms" not in baseline_data:
        raise SystemExit("Baseline missing key: runtime_ms")
    comparisons.append(_compare_metric("runtime_ms", float(baseline_data["runtime_ms"]), runtime_ms, tolerance=ENVIRONMENT_TOLERANCE))

    args.output.write_text(json.dumps({"comparisons": comparisons}, indent=2, sort_keys=True))

    failures = [entry for entry in comparisons if entry["status"] == "fail"]
    if failures:
        raise SystemExit(f"Regression detected in {len(failures)} metrics")


if __name__ == "__main__":
    main()

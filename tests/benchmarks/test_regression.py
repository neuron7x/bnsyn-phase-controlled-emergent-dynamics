from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.metrics import metrics_to_dict, run_benchmark
from benchmarks.scenarios import get_scenario_by_name

BASELINE_DIR = Path("benchmarks/baselines")


def _load_baseline(name: str) -> dict[str, float]:
    path = BASELINE_DIR / f"{name}.json"
    data = json.loads(path.read_text())
    return data["metrics"]


@pytest.mark.parametrize("scenario_name", ["small_network", "medium_network", "large_network"])
def test_benchmark_regression(scenario_name: str) -> None:
    scenario = get_scenario_by_name(scenario_name)
    metrics = metrics_to_dict(run_benchmark(scenario))
    baseline = _load_baseline(scenario_name)

    for metric_name, bounds in baseline.items():
        value = float(metrics[metric_name])
        min_val = float(bounds["min"])
        max_val = float(bounds["max"])
        assert min_val <= value <= max_val, (
            f"{scenario_name} metric {metric_name}={value:.6f} outside "
            f"[{min_val:.6f}, {max_val:.6f}]"
        )

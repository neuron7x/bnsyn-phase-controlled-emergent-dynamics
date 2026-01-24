from __future__ import annotations

import numpy as np
import pytest

from benchmarks.metrics import metrics_to_dict, run_benchmark
from benchmarks.scenarios import get_scenario_by_name


@pytest.mark.parametrize("scenario_name", ["small_network", "medium_network", "large_network"])
def test_benchmark_regression(scenario_name: str) -> None:
    scenario = get_scenario_by_name(scenario_name)
    metrics_first = metrics_to_dict(run_benchmark(scenario))
    metrics_second = metrics_to_dict(run_benchmark(scenario))

    for metric_name in metrics_first:
        if metric_name.startswith("performance_"):
            continue
        first = float(metrics_first[metric_name])
        second = float(metrics_second[metric_name])
        assert np.allclose(first, second, rtol=1e-10, atol=1e-12), (
            f"{scenario_name} metric {metric_name} drifted: {first:.12f} vs {second:.12f}"
        )

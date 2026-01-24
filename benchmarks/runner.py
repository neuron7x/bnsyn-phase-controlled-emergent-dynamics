"""Deterministic benchmark runner for BN-Syn performance measurements."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import resource
from bnsyn.config import AdExParams, CriticalityParams, SynapseParams
from bnsyn.rng import seed_all
from bnsyn.sim.network import Network, NetworkParams

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _load_scenarios_module() -> Any:
    module_path = Path(__file__).with_name("scenarios.py")
    spec = importlib.util.spec_from_file_location("benchmarks.deterministic_scenarios", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load deterministic scenarios module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SCENARIOS_MODULE = _load_scenarios_module()
ScenarioSpec = _SCENARIOS_MODULE.ScenarioSpec
get_suite = _SCENARIOS_MODULE.get_suite


@dataclass(frozen=True)
class ScenarioRunMetrics:
    wall_time_sec: float
    peak_rss_mb: float
    per_step_ms: float


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    scenario_name: str
    variant: str
    seed: int
    steps: int
    dt_ms: float
    n_neurons: int
    p_conn: float
    frac_inhib: float
    runs: list[ScenarioRunMetrics]

    def median_metrics(self) -> ScenarioRunMetrics:
        return ScenarioRunMetrics(
            wall_time_sec=statistics.median(run.wall_time_sec for run in self.runs),
            peak_rss_mb=statistics.median(run.peak_rss_mb for run in self.runs),
            per_step_ms=statistics.median(run.per_step_ms for run in self.runs),
        )


def set_deterministic_env() -> None:
    """Pin thread-related environment variables for deterministic timing."""
    for key in THREAD_ENV_VARS:
        os.environ.setdefault(key, "1")
    os.environ.setdefault("PYTHONHASHSEED", "0")


def get_git_sha() -> str:
    """Return the current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _rss_to_mb(rss_kb: float) -> float:
    return float(rss_kb / 1024.0)


def run_scenario_once(spec: ScenarioSpec, seed_offset: int = 0) -> ScenarioRunMetrics:
    """Run a single scenario once and return timing + memory metrics."""
    set_deterministic_env()
    pack = seed_all(spec.seed + seed_offset)
    rng = pack.np_rng

    network = Network(
        NetworkParams(N=spec.n_neurons, p_conn=spec.p_conn, frac_inhib=spec.frac_inhib),
        AdExParams(),
        SynapseParams(),
        CriticalityParams(),
        dt_ms=spec.dt_ms,
        rng=rng,
    )

    start_time = time.perf_counter()
    for _ in range(spec.steps):
        network.step()
    wall_time = time.perf_counter() - start_time

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = _rss_to_mb(rss_kb)
    per_step_ms = (wall_time * 1000.0 / spec.steps) if spec.steps else 0.0
    return ScenarioRunMetrics(
        wall_time_sec=wall_time,
        peak_rss_mb=peak_rss_mb,
        per_step_ms=per_step_ms,
    )


def _expand_scenario(spec: ScenarioSpec) -> list[tuple[ScenarioSpec, str, int]]:
    if not spec.sweep_sizes:
        return [(spec, "base", 0)]
    expanded: list[tuple[ScenarioSpec, str, int]] = []
    for idx, size in enumerate(spec.sweep_sizes):
        expanded_spec = ScenarioSpec(
            scenario_id=spec.scenario_id,
            name=spec.name,
            seed=spec.seed,
            steps=spec.steps,
            dt_ms=spec.dt_ms,
            n_neurons=size,
            p_conn=spec.p_conn,
            frac_inhib=spec.frac_inhib,
            sweep_sizes=None,
        )
        expanded.append((expanded_spec, f"N{size}", idx))
    return expanded


def run_suite(
    suite: str,
    warmup_runs: int = 1,
    measured_runs: int = 3,
) -> dict[str, Any]:
    """Run a benchmark suite and return structured results."""
    set_deterministic_env()
    scenario_specs = get_suite(suite)
    results: list[ScenarioResult] = []

    for spec in scenario_specs:
        for expanded_spec, variant, seed_offset in _expand_scenario(spec):
            for _ in range(warmup_runs):
                run_scenario_once(expanded_spec, seed_offset=seed_offset)
            runs = [
                run_scenario_once(expanded_spec, seed_offset=seed_offset)
                for _ in range(measured_runs)
            ]
            results.append(
                ScenarioResult(
                    scenario_id=expanded_spec.scenario_id,
                    scenario_name=expanded_spec.name,
                    variant=variant,
                    seed=expanded_spec.seed + seed_offset,
                    steps=expanded_spec.steps,
                    dt_ms=expanded_spec.dt_ms,
                    n_neurons=expanded_spec.n_neurons,
                    p_conn=expanded_spec.p_conn,
                    frac_inhib=expanded_spec.frac_inhib,
                    runs=runs,
                )
            )

    payload = {
        "metadata": collect_metadata(suite),
        "results": [scenario_result_to_dict(result) for result in results],
    }
    return payload


def scenario_result_to_dict(result: ScenarioResult) -> dict[str, Any]:
    median = result.median_metrics()
    return {
        "scenario_id": result.scenario_id,
        "scenario_name": result.scenario_name,
        "variant": result.variant,
        "seed": result.seed,
        "steps": result.steps,
        "dt_ms": result.dt_ms,
        "n_neurons": result.n_neurons,
        "p_conn": result.p_conn,
        "frac_inhib": result.frac_inhib,
        "runs": [
            {
                "wall_time_sec": run.wall_time_sec,
                "peak_rss_mb": run.peak_rss_mb,
                "per_step_ms": run.per_step_ms,
            }
            for run in result.runs
        ],
        "median": {
            "wall_time_sec": median.wall_time_sec,
            "peak_rss_mb": median.peak_rss_mb,
            "per_step_ms": median.per_step_ms,
        },
    }


def collect_metadata(suite: str) -> dict[str, Any]:
    return {
        "suite": suite,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "git_sha": get_git_sha(),
    }


def write_json_report(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

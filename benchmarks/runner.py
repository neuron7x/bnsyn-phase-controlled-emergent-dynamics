"""Deterministic benchmark runner for BN-Syn."""

from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.scenarios import BenchmarkScenario, get_scenarios


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class RunMetrics:
    wall_time_sec: float
    peak_rss_mb: float | None
    per_step_ms: float
    neuron_steps_per_sec: float
    spike_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_time_sec": self.wall_time_sec,
            "peak_rss_mb": self.peak_rss_mb,
            "per_step_ms": self.per_step_ms,
            "neuron_steps_per_sec": self.neuron_steps_per_sec,
            "spike_count": self.spike_count,
        }


def set_deterministic_env() -> None:
    """Limit thread-level nondeterminism by forcing single-threaded BLAS."""
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, "1")


def get_git_sha() -> str:
    """Best-effort git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_cpu_info() -> str:
    """Best-effort CPU info."""
    info = platform.processor() or "unknown"
    if Path("/proc/cpuinfo").exists():
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if "model name" in line:
                        info = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return info


def get_system_metadata() -> dict[str, Any]:
    """Collect reproducibility metadata."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": get_cpu_info(),
        "cpu_count": os.cpu_count(),
    }


def _rss_mb() -> float | None:
    if sys.platform.startswith("linux"):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return None


def run_single(scenario: BenchmarkScenario) -> RunMetrics:
    """Run a single scenario once and return metrics."""
    import gc

    from bnsyn.config import AdExParams, CriticalityParams, SynapseParams
    from bnsyn.rng import seed_all
    from bnsyn.sim.network import Network, NetworkParams

    set_deterministic_env()
    rng_pack = seed_all(scenario.seed)

    gc.collect()

    nparams = NetworkParams(
        N=scenario.N_neurons,
        p_conn=scenario.p_conn,
        frac_inhib=scenario.frac_inhib,
    )
    network = Network(
        nparams,
        AdExParams(),
        SynapseParams(),
        CriticalityParams(),
        dt_ms=scenario.dt_ms,
        rng=rng_pack.np_rng,
    )

    start_time = time.perf_counter()
    spike_count = 0
    for _ in range(scenario.steps):
        metrics = network.step()
        spike_count += int(metrics["A_t1"])
    wall_time = time.perf_counter() - start_time

    per_step_ms = (wall_time / scenario.steps) * 1000.0
    neuron_steps = scenario.N_neurons * scenario.steps
    neuron_steps_per_sec = neuron_steps / wall_time
    peak_rss_mb = _rss_mb()

    return RunMetrics(
        wall_time_sec=wall_time,
        peak_rss_mb=peak_rss_mb,
        per_step_ms=per_step_ms,
        neuron_steps_per_sec=neuron_steps_per_sec,
        spike_count=spike_count,
    )


def summarize_runs(runs: list[RunMetrics]) -> dict[str, Any]:
    """Summarize run metrics using median statistics."""
    if not runs:
        return {}

    def median(values: list[float]) -> float:
        return float(statistics.median(values))

    wall_times = [run.wall_time_sec for run in runs]
    per_step = [run.per_step_ms for run in runs]
    throughput = [run.neuron_steps_per_sec for run in runs]
    peaks = [run.peak_rss_mb for run in runs if run.peak_rss_mb is not None]

    summary: dict[str, Any] = {
        "wall_time_sec_median": median(wall_times),
        "per_step_ms_median": median(per_step),
        "neuron_steps_per_sec_median": median(throughput),
        "spike_count_total": sum(run.spike_count for run in runs),
    }
    summary["peak_rss_mb_median"] = median(peaks) if peaks else None
    return summary


def run_suite(
    suite: str,
    warmup_runs: int = 1,
    measured_runs: int = 3,
) -> dict[str, Any]:
    """Run a benchmark suite with deterministic settings."""
    set_deterministic_env()
    scenarios = get_scenarios(suite)
    metadata = get_system_metadata()

    results = []
    for scenario in scenarios:
        for _ in range(warmup_runs):
            run_single(scenario)
        runs = [run_single(scenario) for _ in range(measured_runs)]
        summary = summarize_runs(runs)
        results.append(
            {
                "scenario": scenario.to_dict(),
                "runs": [run.to_dict() for run in runs],
                "summary": summary,
            }
        )

    return {"metadata": metadata, "suite": suite, "results": results}


def write_json(result: dict[str, Any], output_path: str | Path) -> Path:
    """Write benchmark results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return output_path

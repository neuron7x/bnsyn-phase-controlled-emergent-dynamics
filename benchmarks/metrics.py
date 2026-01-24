"""Metrics collection for BN-Syn benchmarks.

Provides timing, memory (RSS peak), and derived throughput metrics.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import psutil


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Metrics collected from a single benchmark run."""

    wall_time_sec: float
    peak_rss_mb: float
    per_step_ms: float
    neuron_steps_per_sec: float
    spike_count: int
    params: dict[str, Any]


class MetricsCollector:
    """Context manager for collecting benchmark metrics."""

    def __init__(self, n_neurons: int, n_steps: int, params: dict[str, Any]) -> None:
        self.n_neurons = n_neurons
        self.n_steps = n_steps
        self.params = params
        self.process = psutil.Process(os.getpid())
        self._start_time = 0.0
        self._start_rss = 0.0
        self._spike_count = 0

    def __enter__(self) -> MetricsCollector:
        # Force GC and get baseline RSS
        import gc

        gc.collect()
        self._start_rss = self.process.memory_info().rss / (1024 * 1024)  # MB
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def record_spikes(self, count: int) -> None:
        """Record spike count from simulation."""
        self._spike_count += count

    def finalize(self) -> BenchmarkMetrics:
        """Finalize and return metrics."""
        wall_time = time.perf_counter() - self._start_time
        current_rss = self.process.memory_info().rss / (1024 * 1024)  # MB
        peak_rss = max(current_rss, self._start_rss)

        per_step_ms = (wall_time / self.n_steps) * 1000.0
        neuron_steps = self.n_neurons * self.n_steps
        neuron_steps_per_sec = neuron_steps / wall_time

        return BenchmarkMetrics(
            wall_time_sec=wall_time,
            peak_rss_mb=peak_rss,
            per_step_ms=per_step_ms,
            neuron_steps_per_sec=neuron_steps_per_sec,
            spike_count=self._spike_count,
            params=self.params,
        )


def format_metrics(metrics: BenchmarkMetrics) -> str:
    """Format metrics for human-readable output."""
    return (
        f"wall_time: {metrics.wall_time_sec:.3f}s, "
        f"peak_rss: {metrics.peak_rss_mb:.1f}MB, "
        f"per_step: {metrics.per_step_ms:.3f}ms, "
        f"throughput: {metrics.neuron_steps_per_sec:.0f} neuron-steps/sec, "
        f"spikes: {metrics.spike_count}"
    )

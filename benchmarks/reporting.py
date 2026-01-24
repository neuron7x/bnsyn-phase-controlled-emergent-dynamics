"""Reporting utilities for BN-Syn benchmarks."""

from __future__ import annotations

from typing import Any


def _format_optional(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def render_table(report: dict[str, Any]) -> str:
    """Render a simple human-readable table for benchmark results."""
    lines = []
    metadata = report["metadata"]
    lines.append("BN-Syn Benchmark Summary")
    lines.append(f"Git SHA: {metadata['git_sha']}")
    lines.append(f"Python: {metadata['python_version']}")
    lines.append(f"Platform: {metadata['platform']}")
    lines.append(f"CPU: {metadata['cpu']}")
    lines.append("")
    header = (
        f"{'Scenario':<20} {'N':>6} {'Steps':>7} {'dt(ms)':>7} "
        f"{'Wall(s)':>8} {'RSS(MB)':>8} {'Step(ms)':>9} {'Throughput':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for entry in report["results"]:
        scenario = entry["scenario"]
        summary = entry["summary"]
        lines.append(
            f"{scenario['name']:<20} "
            f"{scenario['N_neurons']:>6} "
            f"{scenario['steps']:>7} "
            f"{scenario['dt_ms']:>7.2f} "
            f"{summary['wall_time_sec_median']:>8.3f} "
            f"{_format_optional(summary['peak_rss_mb_median'], 1):>8} "
            f"{summary['per_step_ms_median']:>9.3f} "
            f"{summary['neuron_steps_per_sec_median']:>12.0f}"
        )

    return "\n".join(lines)

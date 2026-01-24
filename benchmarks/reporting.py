"""Reporting utilities for benchmark results."""

from __future__ import annotations

from typing import Any


def format_table(payload: dict[str, Any]) -> str:
    results = payload.get("results", [])
    headers = (
        "Scenario",
        "Variant",
        "Steps",
        "dt_ms",
        "N",
        "Wall_s",
        "RSS_MB",
        "Step_ms",
    )
    rows = []
    for result in results:
        median = result["median"]
        rows.append(
            (
                f"{result['scenario_id']} {result['scenario_name']}",
                result["variant"],
                str(result["steps"]),
                f"{result['dt_ms']:.3f}",
                str(result["n_neurons"]),
                f"{median['wall_time_sec']:.4f}",
                f"{median['peak_rss_mb']:.1f}",
                f"{median['per_step_ms']:.4f}",
            )
        )

    columns = list(zip(headers, *rows)) if rows else [(header,) for header in headers]
    widths = [max(len(str(item)) for item in column) for column in columns]

    def format_row(items: tuple[str, ...]) -> str:
        return " | ".join(item.ljust(width) for item, width in zip(items, widths))

    lines = [format_row(headers)]
    lines.append("-+-".join("-" * width for width in widths))
    for row in rows:
        lines.append(format_row(row))
    return "\n".join(lines)

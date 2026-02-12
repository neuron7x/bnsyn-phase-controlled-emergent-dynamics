"""Unit tests for CLI commands to improve coverage.

Parameters
----------
None

Returns
-------
None

Notes
-----
Direct unit tests for CLI command functions.

References
----------
docs/sleep_stack.md
"""

from __future__ import annotations

import argparse
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

from bnsyn.cli import (
    _build_sleep_stack_metrics,
    _cmd_demo,
    _cmd_dtcheck,
    _cmd_sleep_stack,
    _scaled_sleep_stages,
)


def test_cmd_demo_direct() -> None:
    """Test _cmd_demo function directly."""
    args = argparse.Namespace(
        steps=50,
        dt_ms=0.1,
        seed=42,
        N=40,
    )
    result = _cmd_demo(args)
    assert result == 0


def test_cmd_dtcheck_direct() -> None:
    """Test _cmd_dtcheck function directly."""
    args = argparse.Namespace(
        steps=50,
        dt_ms=0.1,
        dt2_ms=0.05,
        seed=42,
        N=40,
    )
    result = _cmd_dtcheck(args)
    assert result == 0


def test_cmd_sleep_stack_direct() -> None:
    """Test _cmd_sleep_stack function directly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "test_output"

        args = argparse.Namespace(
            seed=123,
            N=64,
            backend="reference",
            steps_wake=50,
            steps_sleep=50,
            out=str(out_dir),
        )

        result = _cmd_sleep_stack(args)
        assert result == 0

        # Verify outputs
        manifest_path = out_dir / "manifest.json"
        metrics_path = out_dir / "metrics.json"

        assert manifest_path.exists()
        assert metrics_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["seed"] == 123
        assert manifest["steps_wake"] == 50

        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "wake" in metrics
        assert "sleep" in metrics
        assert "transitions" in metrics
        assert "attractors" in metrics
        assert "consolidation" in metrics


def test_cmd_sleep_stack_with_custom_sleep_duration() -> None:
    """Test _cmd_sleep_stack with non-default sleep duration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "test_output2"

        args = argparse.Namespace(
            seed=456,
            N=64,
            backend="reference",
            steps_wake=30,
            steps_sleep=300,  # Different from default 600
            out=str(out_dir),
        )

        result = _cmd_sleep_stack(args)
        assert result == 0

        # Verify that manifest records the correct steps
        manifest_path = out_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["steps_sleep"] == 300


def test_scaled_sleep_stages_matches_default_for_600_steps() -> None:
    """Ensure default sleep-stage schedule is preserved for canonical 600 steps."""
    from bnsyn.sleep import default_human_sleep_cycle

    default_stages = default_human_sleep_cycle()
    scaled_stages = _scaled_sleep_stages(600)

    assert len(scaled_stages) == len(default_stages)
    assert [s.stage for s in scaled_stages] == [s.stage for s in default_stages]
    assert [s.duration_steps for s in scaled_stages] == [s.duration_steps for s in default_stages]


def test_scaled_sleep_stages_scales_nondefault_budget() -> None:
    """Ensure non-default sleep budget rescales durations deterministically."""
    from bnsyn.sleep import default_human_sleep_cycle

    target_steps = 300
    scale = target_steps / 450
    default_stages = default_human_sleep_cycle()
    scaled_stages = _scaled_sleep_stages(target_steps)

    expected_durations = [int(stage.duration_steps * scale) for stage in default_stages]
    assert [stage.duration_steps for stage in scaled_stages] == expected_durations


def test_build_sleep_stack_metrics_contains_expected_fields() -> None:
    """Ensure metrics builder keeps expected schema and deterministic values."""
    wake_metrics = [
        {"sigma": 1.2, "spike_rate_hz": 6.0},
        {"sigma": 0.8, "spike_rate_hz": 4.0},
    ]
    transition = SimpleNamespace(
        step=10,
        from_phase=SimpleNamespace(name="WAKE"),
        to_phase=SimpleNamespace(name="SLEEP"),
        sigma_before=1.2,
        sigma_after=0.9,
        sharpness=0.3,
    )
    cryst_state = SimpleNamespace(progress=0.4, phase=SimpleNamespace(name="GROWTH"))

    metrics = _build_sleep_stack_metrics(
        backend="reference",
        steps_wake=2,
        wake_metrics=wake_metrics,
        sleep_summary={"total_steps": 123},
        transitions=[transition],
        attractors=[object(), object()],
        cryst_state=cryst_state,
        cons_stats={"count": 2, "consolidated_count": 1},
        memory_count=7,
    )

    assert metrics["backend"] == "reference"
    assert metrics["wake"]["steps"] == 2
    assert metrics["wake"]["mean_sigma"] == 1.0
    assert metrics["wake"]["mean_spike_rate"] == 5.0
    assert metrics["wake"]["memories_recorded"] == 7
    assert metrics["attractors"]["count"] == 2
    assert metrics["attractors"]["crystallization_progress"] == 0.4
    assert metrics["attractors"]["phase"] == "GROWTH"
    assert metrics["transitions"][0]["from"] == "WAKE"
    assert metrics["transitions"][0]["to"] == "SLEEP"

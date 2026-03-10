from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np

from bnsyn.experiments.declarative import _build_phase_space_report, run_canonical_live_bundle


def test_phase_space_handcrafted_trace_is_deterministic() -> None:
    rates = np.asarray([0.0, 1.0, 2.0, 1.0], dtype=np.float64)
    sigmas = np.asarray([1.0, 2.0, 3.0, 2.0], dtype=np.float64)

    report_a = _build_phase_space_report(
        seed=7,
        n_neurons=4,
        dt_ms=1.0,
        duration_ms=4.0,
        steps=4,
        rate_trace_hz=rates,
        sigma_trace=sigmas,
    )
    report_b = _build_phase_space_report(
        seed=7,
        n_neurons=4,
        dt_ms=1.0,
        duration_ms=4.0,
        steps=4,
        rate_trace_hz=rates,
        sigma_trace=sigmas,
    )

    assert report_a == report_b
    assert report_a["point_count"] == 4


def test_phase_space_correlation_correctness() -> None:
    rates = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    sigmas = np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    report = _build_phase_space_report(
        seed=1,
        n_neurons=2,
        dt_ms=0.5,
        duration_ms=2.0,
        steps=4,
        rate_trace_hz=rates,
        sigma_trace=sigmas,
    )
    assert abs(float(report["rate_sigma_correlation"]) - 1.0) < 1e-12


def test_phase_space_report_schema_and_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_dir)

    phase_path = out_dir / "phase_space_report.json"
    manifest_path = out_dir / "run_manifest.json"
    assert phase_path.exists()

    report = json.loads(phase_path.read_text(encoding="utf-8"))
    assert set(report.keys()) == {
        "schema_version",
        "seed",
        "N",
        "dt_ms",
        "duration_ms",
        "steps",
        "state_axes",
        "point_count",
        "rate_mean_hz",
        "sigma_mean",
        "rate_sigma_correlation",
        "trajectory_length_l2",
        "bounding_box",
        "centroid",
        "occupied_cell_fraction",
    }
    assert report["state_axes"] == ["population_rate_hz", "sigma"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "phase_space_report.json" in manifest["artifacts"]


def test_phase_space_report_deterministic_repeated_runs(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_a)
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_b)

    bytes_a = (out_a / "phase_space_report.json").read_bytes()
    bytes_b = (out_b / "phase_space_report.json").read_bytes()
    assert bytes_a == bytes_b


def test_phase_space_fail_closed_on_steps_mismatch() -> None:
    rates = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    sigmas = np.asarray([1.0, 1.1, 1.2], dtype=np.float64)
    try:
        _build_phase_space_report(
            seed=1,
            n_neurons=2,
            dt_ms=0.5,
            duration_ms=1.5,
            steps=4,
            rate_trace_hz=rates,
            sigma_trace=sigmas,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for steps mismatch")


def test_phase_space_report_schema_contract() -> None:
    rates = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    sigmas = np.asarray([1.0, 1.1, 1.2, 1.3], dtype=np.float64)
    report = _build_phase_space_report(
        seed=1,
        n_neurons=2,
        dt_ms=0.5,
        duration_ms=2.0,
        steps=4,
        rate_trace_hz=rates,
        sigma_trace=sigmas,
    )
    schema = json.loads(Path("schemas/phase-space-report.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(instance=report, schema=schema)
    assert report["point_count"] == len(rates)


def test_phase_space_fail_closed_on_trace_length_mismatch() -> None:
    rates = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    sigmas = np.asarray([1.0, 1.1], dtype=np.float64)
    try:
        _build_phase_space_report(
            seed=1,
            n_neurons=2,
            dt_ms=0.5,
            duration_ms=1.0,
            steps=3,
            rate_trace_hz=rates,
            sigma_trace=sigmas,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for trace length mismatch")


def test_phase_space_zero_and_constant_traces_cover_edge_branches() -> None:
    # constant traces exercise zero-variance correlation branch and collapsed bbox indexing branch
    rates = np.asarray([5.0, 5.0, 5.0], dtype=np.float64)
    sigmas = np.asarray([1.2, 1.2, 1.2], dtype=np.float64)
    report = _build_phase_space_report(
        seed=9,
        n_neurons=3,
        dt_ms=1.0,
        duration_ms=3.0,
        steps=3,
        rate_trace_hz=rates,
        sigma_trace=sigmas,
    )
    assert report["rate_sigma_correlation"] == 0.0
    assert report["trajectory_length_l2"] == 0.0
    assert report["occupied_cell_fraction"] == 1.0 / (32.0 * 32.0)


def test_phase_space_empty_trace_edge_branch() -> None:
    rates = np.asarray([], dtype=np.float64)
    sigmas = np.asarray([], dtype=np.float64)
    report = _build_phase_space_report(
        seed=11,
        n_neurons=1,
        dt_ms=1.0,
        duration_ms=0.0,
        steps=0,
        rate_trace_hz=rates,
        sigma_trace=sigmas,
    )
    assert report["point_count"] == 0
    assert report["trajectory_length_l2"] == 0.0
    assert report["occupied_cell_fraction"] == 0.0

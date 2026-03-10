from __future__ import annotations

import json
from pathlib import Path

import pytest

from bnsyn.experiments.declarative import run_canonical_live_bundle


def test_canonical_live_bundle_writes_required_outputs(tmp_path: Path) -> None:
    bundle = run_canonical_live_bundle(
        "configs/canonical_profile.yaml",
        artifact_dir=tmp_path / "canonical_run",
    )

    out_dir = Path(str(bundle["artifact_dir"]))
    assert out_dir == tmp_path / "canonical_run"

    summary_path = out_dir / "summary_metrics.json"
    manifest_path = out_dir / "run_manifest.json"
    criticality_report_path = out_dir / "criticality_report.json"
    emergence_plot_path = out_dir / "emergence_plot.png"
    raster_path = out_dir / "raster_plot.png"
    rate_plot_path = out_dir / "population_rate_plot.png"
    assert summary_path.exists()
    assert manifest_path.exists()
    assert criticality_report_path.exists()
    assert emergence_plot_path.exists()
    assert raster_path.exists()
    assert rate_plot_path.exists()

    metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {
        "spike_events",
        "rate_mean_hz",
        "rate_peak_hz",
        "rate_variance",
        "sigma_mean",
        "sigma_final",
        "sigma_variance",
        "steps",
        "dt_ms",
        "duration_ms",
    }
    assert required.issubset(metrics)
    assert metrics["spike_events"] > 0
    assert metrics["rate_mean_hz"] > 0.0
    assert metrics["rate_variance"] > 0.0

    criticality = json.loads(criticality_report_path.read_text(encoding="utf-8"))
    criticality_required = {
        "schema_version", "seed", "N", "dt_ms", "duration_ms", "steps",
        "sigma_mean", "sigma_final", "sigma_variance", "rate_mean_hz",
        "rate_peak_hz", "spike_events", "sigma_distance_from_1",
        "sigma_within_band_fraction", "active_steps_fraction",
        "nonzero_rate_steps_fraction", "burstiness_proxy", "rate_cv",
    }
    assert criticality_required.issubset(criticality)


def test_canonical_live_bundle_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_a)
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_b)

    summary_a = json.loads((out_a / "summary_metrics.json").read_text(encoding="utf-8"))
    summary_b = json.loads((out_b / "summary_metrics.json").read_text(encoding="utf-8"))
    criticality_a = json.loads((out_a / "criticality_report.json").read_text(encoding="utf-8"))
    criticality_b = json.loads((out_b / "criticality_report.json").read_text(encoding="utf-8"))
    assert summary_a == summary_b
    assert criticality_a == criticality_b


def test_cli_run_profile_canonical_end_to_end(monkeypatch, tmp_path: Path) -> None:
    from bnsyn import cli

    monkeypatch.setattr(
        "sys.argv",
        ["bnsyn", "run", "--profile", "canonical", "--output", str(tmp_path / "canonical_run")],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0

    summary_path = tmp_path / "canonical_run" / "summary_metrics.json"
    assert summary_path.exists()
    metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    assert metrics["spike_events"] > 0

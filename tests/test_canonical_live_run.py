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
    raster_path = out_dir / "raster_plot.png"
    rate_plot_path = out_dir / "population_rate_plot.png"
    assert summary_path.exists()
    assert raster_path.exists()
    assert rate_plot_path.exists()

    metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {
        "spike_events",
        "rate_mean_hz",
        "rate_variance",
        "sigma_mean",
        "sigma_variance",
    }
    assert required.issubset(metrics)
    assert metrics["spike_events"] > 0
    assert metrics["rate_mean_hz"] > 0.0
    assert metrics["rate_variance"] > 0.0


def test_canonical_live_bundle_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_a)
    run_canonical_live_bundle("configs/canonical_profile.yaml", artifact_dir=out_b)

    summary_a = json.loads((out_a / "summary_metrics.json").read_text(encoding="utf-8"))
    summary_b = json.loads((out_b / "summary_metrics.json").read_text(encoding="utf-8"))
    assert summary_a == summary_b


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

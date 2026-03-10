"""Tests for emergence run/sweep/plot reproducibility and contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from bnsyn import cli
from bnsyn.experiments import declarative
from bnsyn.schemas.experiment import BNSynExperimentConfig
from bnsyn.sim.network import run_simulation
from bnsyn.viz.emergence_plot import plot_emergence_npz


def test_run_simulation_writes_npz_artifact_contract(tmp_path: Path) -> None:
    metrics = run_simulation(
        steps=10,
        dt_ms=0.1,
        seed=7,
        N=20,
        external_current_pA=100.0,
        artifact_dir=tmp_path,
    )
    assert metrics["sigma_mean"] >= 0.0
    npz_path = tmp_path / "run_7.npz"
    assert npz_path.exists()

    with np.load(npz_path) as data:
        assert set(data.files) == {
            "format_version",
            "spike_steps",
            "spike_neurons",
            "sigma_trace",
            "rate_trace_hz",
            "dt_ms",
            "steps",
            "N",
            "seed",
            "external_current_pA",
        }
        assert int(data["format_version"]) == 1
        assert int(data["steps"]) == 10
        assert data["rate_trace_hz"].shape[0] == 10
        assert data["sigma_trace"].shape[0] == 10


def test_emergence_plot_from_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_simulation(
        steps=10,
        dt_ms=0.1,
        seed=8,
        N=20,
        external_current_pA=120.0,
        artifact_dir=tmp_path,
    )

    class _FakeFigure:
        def tight_layout(self) -> None:
            return None

        def savefig(self, out_path: Path, dpi: int) -> None:
            _ = dpi
            out_path.write_bytes(b"fake")

    class _FakeAxes:
        def scatter(self, *_args: object, **_kwargs: object) -> None:
            return None

        def plot(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_ylabel(self, _label: str) -> None:
            return None

        def set_title(self, _title: str) -> None:
            return None

        def set_xlabel(self, _label: str) -> None:
            return None

    class _FakePlt:
        def subplots(
            self, _rows: int, _cols: int, figsize: tuple[int, int], sharex: bool
        ) -> tuple[_FakeFigure, list[_FakeAxes]]:
            _ = (figsize, sharex)
            return _FakeFigure(), [_FakeAxes(), _FakeAxes(), _FakeAxes()]

        def close(self, _fig: _FakeFigure) -> None:
            return None

    monkeypatch.setattr("bnsyn.viz.emergence_plot._load_pyplot", lambda: _FakePlt())
    out = tmp_path / "emergence.png"
    plot_emergence_npz(tmp_path / "run_8.npz", out)
    assert out.exists()


def test_emergence_plot_reports_missing_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bnsyn.viz.emergence_plot.importlib.import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("missing matplotlib")),
    )
    monkeypatch.setattr("bnsyn.viz.emergence_plot._plt", None)
    with pytest.raises(RuntimeError, match="Visualization requires matplotlib"):
        import bnsyn.viz.emergence_plot as emergence_plot

        emergence_plot._load_pyplot()


def test_emergence_cli_sweep_writes_structured_report(tmp_path: Path) -> None:
    args = argparse.Namespace(N=20, dt_ms=0.1, duration_ms=20.0, seed=42, out=tmp_path)
    rc = cli._cmd_emergence_sweep(args)
    assert rc == 0

    report_path = tmp_path / "emergence_sweep_report.json"
    report = json.loads(report_path.read_text())
    assert report["currents_pA"] == [365.0, 380.0, 395.0, 410.0, 450.0]
    assert report["steps"] == 200
    assert len(report["runs"]) == 5
    artifact_paths = [Path(run["artifact_npz"]) for run in report["runs"]]
    assert len(set(artifact_paths)) == 5
    for path in artifact_paths:
        assert path.exists()


def test_emergence_cli_rejects_non_integral_step_count(tmp_path: Path) -> None:
    args = argparse.Namespace(N=20, dt_ms=0.3, duration_ms=20.0, seed=42, out=tmp_path)
    with pytest.raises(ValueError, match="duration-ms must be an integer multiple of dt-ms"):
        cli._cmd_emergence_sweep(args)


def test_external_current_trend_sanity() -> None:
    low = run_simulation(steps=50, dt_ms=0.1, seed=42, N=60, external_current_pA=365.0)
    high = run_simulation(steps=50, dt_ms=0.1, seed=42, N=60, external_current_pA=450.0)
    assert high["rate_mean_hz"] >= low["rate_mean_hz"]


def test_declarative_passes_external_current_and_artifact_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = BNSynExperimentConfig(
        experiment={"name": "emergence", "version": "v1", "seeds": [1]},
        network={"size": 20},
        simulation={
            "duration_ms": 1.0,
            "dt_ms": 0.1,
            "external_current_pA": 410.0,
            "artifact_dir": "artifacts/test",
        },
    )
    calls: list[dict[str, object]] = []

    def fake_run_simulation(**kwargs: object) -> dict[str, float]:
        calls.append(kwargs)
        return {"sigma_mean": 1.0, "rate_mean_hz": 1.0, "sigma_std": 0.0, "rate_std": 0.0}

    monkeypatch.setattr(declarative, "run_simulation", fake_run_simulation)
    declarative.run_experiment(cfg)
    assert calls[0]["external_current_pA"] == 410.0
    assert calls[0]["artifact_dir"] == "artifacts/test"

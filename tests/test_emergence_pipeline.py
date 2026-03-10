"""Tests for emergence orchestration, CLI, and plot contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from bnsyn import cli
from bnsyn.experiments import declarative
from bnsyn.experiments.emergence import run_emergence_to_disk
from bnsyn.numerics.time import compute_steps_exact
from bnsyn.schemas.experiment import BNSynExperimentConfig
from bnsyn.viz.emergence_plot import plot_emergence_npz


def test_compute_steps_exact_rejects_fractional_steps() -> None:
    with pytest.raises(ValueError, match="Non-integer steps"):
        compute_steps_exact(20.0, 0.3)


def test_run_emergence_to_disk_writes_npz_contract(tmp_path: Path) -> None:
    metrics, artifact_path = run_emergence_to_disk(
        N=20,
        dt_ms=0.1,
        duration_ms=1.0,
        seed=7,
        external_current_pA=100.0,
        output_dir=tmp_path,
    )
    assert metrics["sigma_mean"] >= 0.0

    npz_path = Path(artifact_path)
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
        assert str(data["format_version"]) == "1.1.0"


def test_emergence_plot_from_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, artifact_path = run_emergence_to_disk(
        N=20,
        dt_ms=0.1,
        duration_ms=1.0,
        seed=8,
        external_current_pA=120.0,
        output_dir=tmp_path,
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
    plot_emergence_npz(Path(artifact_path), out)
    assert out.exists()




def test_emergence_plot_rejects_wrong_format_version(tmp_path: Path) -> None:
    artifact = tmp_path / "bad.npz"
    np.savez(
        artifact,
        format_version="0.9.0",
        spike_steps=np.asarray([], dtype=np.int64),
        spike_neurons=np.asarray([], dtype=np.int64),
        sigma_trace=np.asarray([1.0], dtype=np.float64),
        rate_trace_hz=np.asarray([1.0], dtype=np.float64),
        dt_ms=np.float64(0.1),
        steps=np.int64(1),
        N=np.int64(20),
        seed=np.int64(1),
        external_current_pA=np.float64(0.0),
    )
    with pytest.raises(RuntimeError, match="Artifact format mismatch"):
        plot_emergence_npz(artifact, tmp_path / "out.png")


def test_emergence_plot_reports_missing_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bnsyn.viz.emergence_plot.importlib.import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("missing matplotlib")),
    )
    monkeypatch.setattr("bnsyn.viz.emergence_plot._plt", None)
    with pytest.raises(RuntimeError, match="Visualization requires matplotlib"):
        import bnsyn.viz.emergence_plot as emergence_plot

        emergence_plot._load_pyplot()


def test_emergence_cli_sweep_writes_unique_artifacts(tmp_path: Path) -> None:
    args = argparse.Namespace(N=20, dt_ms=0.1, duration_ms=1.0, seed=42, out=tmp_path)
    rc = cli._cmd_emergence_sweep(args)
    assert rc == 0

    report = json.loads((tmp_path / "emergence_sweep_report.json").read_text())
    assert report["currents_pA"] == [365.0, 380.0, 395.0, 410.0, 450.0]
    paths = [Path(run["artifact_npz"]) for run in report["runs"]]
    assert len(paths) == 5
    assert len(set(paths)) == 5
    for path in paths:
        assert path.exists()


def test_declarative_uses_external_current(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = BNSynExperimentConfig(
        experiment={"name": "emergence", "version": "v1", "seeds": [1]},
        network={"size": 20},
        simulation={
            "duration_ms": 1.0,
            "dt_ms": 0.1,
            "external_current_pA": 410.0,
        },
    )
    calls: list[dict[str, object]] = []

    def fake_run_simulation(**kwargs: object) -> dict[str, float]:
        calls.append(kwargs)
        return {"sigma_mean": 1.0, "rate_mean_hz": 1.0, "sigma_std": 0.0, "rate_std": 0.0}

    monkeypatch.setattr(declarative, "run_simulation", fake_run_simulation)
    declarative.run_experiment(cfg)
    assert calls[0]["external_current_pA"] == 410.0
    assert "artifact_dir" not in calls[0]

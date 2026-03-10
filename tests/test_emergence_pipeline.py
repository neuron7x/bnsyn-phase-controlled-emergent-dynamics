from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bnsyn import cli
from bnsyn.experiments.emergence import run_emergence_to_disk
from bnsyn.numerics import compute_steps_exact
from bnsyn.viz.emergence_plot import plot_emergence_npz


def test_compute_steps_exact_rejects_fractional_steps() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        compute_steps_exact(1.25, 0.1)


def test_run_emergence_to_disk_writes_npz_contract(tmp_path: Path) -> None:
    metrics, artifact = run_emergence_to_disk(
        N=20,
        dt_ms=0.1,
        duration_ms=10.0,
        seed=7,
        external_current_pA=410.0,
        output_dir=tmp_path,
    )
    assert set(metrics) == {"sigma_mean", "rate_mean_hz", "sigma_std", "rate_std"}

    artifact_path = Path(artifact)
    assert artifact_path.exists()
    with np.load(artifact_path) as data:
        assert str(data["format_version"].item()) == "1.1.0"
        for key in [
            "spike_steps",
            "spike_neurons",
            "sigma_trace",
            "rate_trace_hz",
            "dt_ms",
            "steps",
            "N",
            "seed",
            "external_current_pA",
        ]:
            assert key in data.files


def _write_npz(path: Path, *, version: str = "1.1.0") -> None:
    np.savez(
        path,
        format_version=np.asarray(version),
        spike_steps=np.asarray([0, 1, 2], dtype=np.int64),
        spike_neurons=np.asarray([1, 2, 3], dtype=np.int64),
        sigma_trace=np.asarray([1.0, 1.1, 0.9], dtype=np.float64),
        rate_trace_hz=np.asarray([5.0, 6.0, 7.0], dtype=np.float64),
        dt_ms=np.asarray(0.1),
        steps=np.asarray(3, dtype=np.int64),
        N=np.asarray(10, dtype=np.int64),
        seed=np.asarray(42, dtype=np.int64),
        external_current_pA=np.asarray(410.0),
    )


def test_plotter_with_matplotlib_shim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    npz_path = tmp_path / "in.npz"
    png_path = tmp_path / "out.png"
    _write_npz(npz_path)

    class _Ax:
        def scatter(self, *args: object, **kwargs: object) -> None:
            return None

        def plot(self, *args: object, **kwargs: object) -> None:
            return None

        def set_ylabel(self, *args: object, **kwargs: object) -> None:
            return None

        def set_xlabel(self, *args: object, **kwargs: object) -> None:
            return None

        def set_title(self, *args: object, **kwargs: object) -> None:
            return None

    class _PLT:
        def subplots(self, *args: object, **kwargs: object) -> tuple[object, list[_Ax]]:
            return object(), [_Ax(), _Ax(), _Ax()]

        def tight_layout(self) -> None:
            return None

        def savefig(self, path: str | Path, **kwargs: object) -> None:
            Path(path).write_bytes(b"png")

        def close(self, fig: object) -> None:
            _ = fig

    plt_shim = _PLT()
    monkeypatch.setitem(sys.modules, "matplotlib", SimpleNamespace(pyplot=plt_shim))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt_shim)

    plot_emergence_npz(npz_path, png_path)
    assert png_path.exists()


def test_plotter_rejects_wrong_format_version(tmp_path: Path) -> None:
    npz_path = tmp_path / "bad.npz"
    _write_npz(npz_path, version="1.0.0")
    with pytest.raises(ValueError, match="Unsupported format_version"):
        plot_emergence_npz(npz_path, tmp_path / "x.png")


def test_plotter_reports_missing_matplotlib(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    npz_path = tmp_path / "in.npz"
    _write_npz(npz_path)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)

    with pytest.raises(RuntimeError, match="Visualization requires matplotlib"):
        plot_emergence_npz(npz_path, tmp_path / "x.png")


def test_cli_emergence_run_writes_report_and_artifact(tmp_path: Path) -> None:
    rc = cli._cmd_emergence_run(
        argparse.Namespace(
            N=24,
            dt_ms=0.1,
            duration_ms=10.0,
            seed=42,
            external_current_pA=410.0,
            out=tmp_path,
        )
    )
    assert rc == 0
    report = json.loads((tmp_path / "emergence_run_report.json").read_text(encoding="utf-8"))
    assert Path(report["artifact_npz"]).exists()


def test_cli_emergence_sweep_writes_unique_artifacts(tmp_path: Path) -> None:
    rc = cli._cmd_emergence_sweep(
        argparse.Namespace(N=24, dt_ms=0.1, duration_ms=10.0, seed=42, out=tmp_path)
    )
    assert rc == 0
    report = json.loads((tmp_path / "emergence_sweep_report.json").read_text(encoding="utf-8"))
    artifacts = [run["artifact_npz"] for run in report["runs"]]
    assert len(artifacts) == len(cli.EMERGENCE_SWEEP_CURRENTS_PA)
    assert len(set(artifacts)) == len(artifacts)
    assert all(Path(path).exists() for path in artifacts)



def test_compute_steps_exact_rejects_nonpositive_dt() -> None:
    with pytest.raises(ValueError, match="dt_ms must be greater than 0"):
        compute_steps_exact(1.0, 0.0)


def test_run_emergence_to_disk_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed must be a positive integer"):
        run_emergence_to_disk(
            N=20,
            dt_ms=0.1,
            duration_ms=10.0,
            seed=0,
            external_current_pA=410.0,
            output_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="external_current_pA must be a finite real number"):
        run_emergence_to_disk(
            N=20,
            dt_ms=0.1,
            duration_ms=10.0,
            seed=1,
            external_current_pA=float("nan"),
            output_dir=tmp_path,
        )


def test_plotter_rejects_missing_required_fields(tmp_path: Path) -> None:
    npz_path = tmp_path / "missing.npz"
    np.savez(npz_path, format_version=np.asarray("1.1.0"))
    with pytest.raises(ValueError, match="Missing required NPZ fields"):
        plot_emergence_npz(npz_path, tmp_path / "x.png")


def test_plotter_rejects_shape_invariant_violation(tmp_path: Path) -> None:
    npz_path = tmp_path / "shape_bad.npz"
    np.savez(
        npz_path,
        format_version=np.asarray("1.1.0"),
        spike_steps=np.asarray([0, 1], dtype=np.int64),
        spike_neurons=np.asarray([1], dtype=np.int64),
        sigma_trace=np.asarray([1.0, 1.1, 0.9], dtype=np.float64),
        rate_trace_hz=np.asarray([5.0, 6.0, 7.0], dtype=np.float64),
        dt_ms=np.asarray(0.1),
        steps=np.asarray(3, dtype=np.int64),
        N=np.asarray(10, dtype=np.int64),
        seed=np.asarray(42, dtype=np.int64),
        external_current_pA=np.asarray(410.0),
    )
    with pytest.raises(ValueError, match="identical shapes"):
        plot_emergence_npz(npz_path, tmp_path / "x.png")

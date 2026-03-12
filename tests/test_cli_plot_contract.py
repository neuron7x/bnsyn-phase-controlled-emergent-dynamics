from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bnsyn.cli import (
    _cmd_plot,
    _cmd_proof_check_determinism,
    _cmd_proof_check_envelope,
    _cmd_proof_evaluate,
    _cmd_proof_validate_bundle,
)


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    src_path = str(Path("src").resolve())
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_path
    )
    return env


def test_cmd_plot_writes_canonical_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "canonical_run"
    args = argparse.Namespace(out=str(out_dir))
    rc = _cmd_plot(args)
    assert rc == 0

    plot_path = out_dir / "emergence_plot.png"
    summary_path = out_dir / "summary_metrics.json"
    manifest_path = out_dir / "run_manifest.json"

    assert plot_path.exists()
    assert summary_path.exists()
    assert manifest_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["seed"] == 123
    assert summary["N"] > 0
    assert "rate_mean_hz" in summary
    assert "sigma_mean" in summary

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["cmd"] == "bnsyn run --profile canonical --plot"
    assert "artifacts" in manifest
    assert "emergence_plot.png" in manifest["artifacts"]
    assert "summary_metrics.json" in manifest["artifacts"]
    assert "criticality_report.json" in manifest["artifacts"]
    assert "avalanche_report.json" in manifest["artifacts"]
    assert "phase_space_report.json" in manifest["artifacts"]
    assert "population_rate_trace.npy" in manifest["artifacts"]
    assert "sigma_trace.npy" in manifest["artifacts"]
    assert "coherence_trace.npy" in manifest["artifacts"]
    assert "phase_space_rate_sigma.png" in manifest["artifacts"]
    assert "phase_space_rate_coherence.png" in manifest["artifacts"]
    assert "phase_space_activity_map.png" in manifest["artifacts"]
    assert manifest["artifacts"]["run_manifest.json"] == "self-unhashed"


def test_cli_plot_runs_and_emits_contract(tmp_path: Path) -> None:
    out_dir = tmp_path / "plot_cli"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bnsyn.cli",
            "plot",
            "--out",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_cli_env(),
    )
    assert proc.returncode == 0, f"plot command failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["artifacts"] == [
        "emergence_plot.png",
        "summary_metrics.json",
        "criticality_report.json",
        "avalanche_report.json",
        "phase_space_report.json",
        "population_rate_trace.npy",
        "sigma_trace.npy",
        "coherence_trace.npy",
        "phase_space_rate_sigma.png",
        "phase_space_rate_coherence.png",
        "phase_space_activity_map.png",
        "avalanche_fit_report.json",
        "robustness_report.json",
        "envelope_report.json",
        "run_manifest.json",
    ]


def test_cmd_plot_returns_error_when_bundle_raises(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    def _boom(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr("bnsyn.experiments.declarative.run_canonical_live_bundle", _boom)
    rc = _cmd_plot(argparse.Namespace(out=str(tmp_path / "ignored")))
    captured = capsys.readouterr()
    assert rc == 1
    assert "Error running canonical compatibility plot wrapper: boom" in captured.out


def test_cmd_proof_evaluate_emits_expected_payload(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    expected_path = tmp_path / "proof_report.json"
    expected = SimpleNamespace(report={"verdict": "PASS", "verdict_code": 0}, report_path=expected_path)
    monkeypatch.setattr("bnsyn.proof.evaluate.evaluate_and_emit", lambda _artifact_dir: expected)

    rc = _cmd_proof_evaluate(SimpleNamespace(artifact_dir=tmp_path))
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload == {
        "status": "ok",
        "artifact_dir": str(tmp_path),
        "proof_report_path": expected_path.as_posix(),
        "verdict": "PASS",
        "verdict_code": 0,
    }


def test_cmd_proof_validate_bundle_returns_zero_on_pass(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    monkeypatch.setattr("bnsyn.proof.bundle_validator.validate_canonical_bundle", lambda _p: {"status": "PASS", "errors": []})
    rc = _cmd_proof_validate_bundle(argparse.Namespace(artifact_dir=tmp_path))
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["status"] == "PASS"


def test_cmd_proof_validate_bundle_returns_nonzero_on_fail(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    monkeypatch.setattr("bnsyn.proof.bundle_validator.validate_canonical_bundle", lambda _p: {"status": "FAIL", "errors": ["x"]})
    rc = _cmd_proof_validate_bundle(argparse.Namespace(artifact_dir=tmp_path))
    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["status"] == "FAIL"


def test_cmd_proof_check_determinism_returns_nonzero_on_fail(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    monkeypatch.setattr("bnsyn.proof.evaluate.evaluate_gate_g6_determinism", lambda _p: {"status": "FAIL", "details": "bad"})
    rc = _cmd_proof_check_determinism(argparse.Namespace(artifact_dir=tmp_path))
    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["status"] == "FAIL"


def test_cmd_proof_check_envelope_returns_nonzero_on_fail(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    monkeypatch.setattr("bnsyn.proof.evaluate.evaluate_gate_g8_repro_envelope", lambda _p: {"status": "FAIL", "details": "bad"})
    rc = _cmd_proof_check_envelope(argparse.Namespace(artifact_dir=tmp_path))
    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["status"] == "FAIL"

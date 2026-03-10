from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from bnsyn.cli import _cmd_plot


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
    assert manifest["cmd"] == "bnsyn run --profile canonical --plot --export-proof"
    assert "artifacts" in manifest
    assert "emergence_plot.png" in manifest["artifacts"]
    assert "summary_metrics.json" in manifest["artifacts"]
    assert "criticality_report.json" in manifest["artifacts"]
    assert "avalanche_report.json" in manifest["artifacts"]
    assert "phase_space_report.json" in manifest["artifacts"]
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
        "run_manifest.json",
    ]

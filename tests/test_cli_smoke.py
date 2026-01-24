import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_demo_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        os.pathsep.join([str(root / "src"), pythonpath]) if pythonpath else str(root / "src")
    )
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "bnsyn.cli",
            "demo",
            "--steps",
            "100",
            "--dt-ms",
            "0.1",
            "--seed",
            "1",
            "--N",
            "80",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    out = json.loads(p.stdout)
    assert "demo" in out


def test_cli_dtcheck_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        os.pathsep.join([str(root / "src"), pythonpath]) if pythonpath else str(root / "src")
    )
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "bnsyn.cli",
            "dtcheck",
            "--steps",
            "50",
            "--dt-ms",
            "0.1",
            "--dt2-ms",
            "0.05",
            "--seed",
            "2",
            "--N",
            "50",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    out = json.loads(p.stdout)
    assert "m_dt" in out
    assert "m_dt2" in out

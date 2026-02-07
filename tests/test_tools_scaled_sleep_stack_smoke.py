from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_scaled_sleep_stack_module_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        os.pathsep.join([str(root / "src"), pythonpath]) if pythonpath else str(root / "src")
    )

    out_dir = tmp_path / "scaled"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bnsyn.tools.run_scaled_sleep_stack",
            "--out",
            str(out_dir),
            "--seed",
            "42",
            "--n",
            "80",
            "--steps-wake",
            "50",
            "--steps-sleep",
            "50",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0

    manifest_path = out_dir / "scaled_run1" / "manifest.json"
    metrics_path = out_dir / "scaled_run1" / "metrics.json"
    assert manifest_path.exists()
    assert metrics_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["seed"] == 42
    assert manifest["N"] == 80
    assert manifest["steps_wake"] == 50
    assert manifest["steps_sleep"] == 50

    metrics = json.loads(metrics_path.read_text())
    assert "backend" in metrics

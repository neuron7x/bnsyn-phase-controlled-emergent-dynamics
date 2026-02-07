from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_scaled_sleep_stack_module_smoke(tmp_path: Path) -> None:
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
            "30",
            "--steps-sleep",
            "30",
            "--baseline-steps-wake",
            "20",
            "--baseline-steps-sleep",
            "10",
            "--determinism-runs",
            "1",
            "--equivalence-steps-wake",
            "20",
            "--skip-backend-equivalence",
            "--no-raster",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0

    manifest_path = out_dir / "scaled_run1" / "manifest.json"
    metrics_path = out_dir / "scaled_run1" / "metrics.json"
    assert manifest_path.exists()
    assert metrics_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["seed"] == 42
    assert manifest["N"] == 80
    assert manifest["steps_wake"] == 30
    assert manifest["steps_sleep"] == 30

    metrics = json.loads(metrics_path.read_text())
    assert "backend" in metrics

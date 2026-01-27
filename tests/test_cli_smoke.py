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


def test_cli_sleep_stack_runs() -> None:
    """Test sleep-stack CLI command."""
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        os.pathsep.join([str(root / "src"), pythonpath]) if pythonpath else str(root / "src")
    )
    
    # Create temp output directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "test_sleep_stack"
        
        subprocess.run(
            [
                sys.executable,
                "-m",
                "bnsyn.cli",
                "sleep-stack",
                "--seed",
                "42",
                "--steps-wake",
                "50",
                "--steps-sleep",
                "50",
                "--out",
                str(out_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        
        # Check that manifest and metrics were created
        manifest_path = out_dir / "manifest.json"
        metrics_path = out_dir / "metrics.json"
        
        assert manifest_path.exists(), "manifest.json not created"
        assert metrics_path.exists(), "metrics.json not created"
        
        # Verify manifest contents
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "seed" in manifest
        assert manifest["seed"] == 42
        assert "steps_wake" in manifest
        assert "steps_sleep" in manifest
        
        # Verify metrics contents
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "wake" in metrics
        assert "sleep" in metrics
        assert "transitions" in metrics
        assert "attractors" in metrics
        assert "consolidation" in metrics
        
        # Check wake metrics
        assert "steps" in metrics["wake"]
        assert "memories_recorded" in metrics["wake"]
        
        # Check consolidation stats
        assert "count" in metrics["consolidation"]
        assert "consolidated_count" in metrics["consolidation"]

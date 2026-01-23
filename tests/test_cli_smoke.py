import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_demo_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), pythonpath]) if pythonpath else str(root / "src")
    p = subprocess.run(
        [sys.executable, "-m", "bnsyn.cli", "demo", "--steps", "100", "--dt-ms", "0.1", "--seed", "1", "--N", "80"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    out = json.loads(p.stdout)
    assert "demo" in out

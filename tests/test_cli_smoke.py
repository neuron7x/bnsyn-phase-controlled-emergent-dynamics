import json
import subprocess
import sys


def test_cli_demo_runs() -> None:
    p = subprocess.run(
        [sys.executable, "-m", "bnsyn.cli", "demo", "--steps", "100", "--dt-ms", "0.1", "--seed", "1", "--N", "80"],
        check=True,
        capture_output=True,
        text=True,
    )
    out = json.loads(p.stdout)
    assert "demo" in out

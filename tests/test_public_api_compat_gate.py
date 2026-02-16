from __future__ import annotations

import subprocess
import sys


def test_public_api_compat_gate_passes() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/check_public_api_compat.py",
            "--baseline",
            "quality/public_api_snapshot.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

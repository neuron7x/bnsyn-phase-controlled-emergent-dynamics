from __future__ import annotations

import subprocess
import sys


def test_validate_ops_slo() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/validate_ops_slo.py", "docs/ops/SLA_SLO.md"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_validate_runbooks() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/validate_runbooks.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

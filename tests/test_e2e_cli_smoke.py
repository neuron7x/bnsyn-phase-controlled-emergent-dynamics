from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.e2e
def test_cli_help_end_to_end() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "bnsyn.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "usage" in completed.stdout.lower()

from __future__ import annotations

import subprocess
import sys


def test_quickstart_consistency_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.check_quickstart_consistency"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PASSED" in result.stdout

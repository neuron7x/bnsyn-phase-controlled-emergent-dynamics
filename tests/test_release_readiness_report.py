from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_release_readiness_report_generates() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_perf_smoke.py", "--output", "quality/perf_results.json"],
        check=True,
    )
    out = Path("quality/release_readiness_report.md")
    proc = subprocess.run(
        [sys.executable, "scripts/release_readiness_report.py", "--output", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "# Release Readiness Report" in text
    assert "| Gate | Status |" in text

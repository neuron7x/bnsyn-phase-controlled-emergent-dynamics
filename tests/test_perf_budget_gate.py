from __future__ import annotations

import subprocess
import sys


def test_perf_budget_gate_passes() -> None:
    run = subprocess.run(
        [sys.executable, "scripts/run_perf_smoke.py", "--output", "quality/perf_results.json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr

    check = subprocess.run(
        [
            sys.executable,
            "scripts/check_perf_budget.py",
            "--baseline",
            "quality/perf_baseline.json",
            "--budgets",
            "quality/perf_budgets.yml",
            "--results",
            "quality/perf_results.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr

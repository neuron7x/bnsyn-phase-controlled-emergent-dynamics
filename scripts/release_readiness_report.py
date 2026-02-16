from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


GATES = [
    ("Tests", "python -m pytest -m \"not validation\" -q"),
    ("Lint", "ruff check ."),
    ("Typecheck", "mypy src --strict --config-file pyproject.toml"),
    ("Ops SLO doc", "python scripts/validate_ops_slo.py docs/ops/SLA_SLO.md"),
    ("Runbooks", "python scripts/validate_runbooks.py"),
    (
        "API compat",
        "python scripts/check_public_api_compat.py --baseline quality/public_api_snapshot.json",
    ),
    (
        "Perf budget",
        "python scripts/check_perf_budget.py --baseline quality/perf_baseline.json --budgets quality/perf_budgets.yml --results quality/perf_results.json",
    ),
]


def _project_version() -> str:
    for line in Path("pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("version ="):
            return line.split("=", 1)[1].strip().strip('"')
    return "unknown"


def _run(cmd: str) -> tuple[int, str]:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)  # nosec B602
    output = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode, output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lines = ["# Release Readiness Report", "", f"- Version: `{_project_version()}`", ""]
    lines.append("## Gate status")
    lines.append("| Gate | Status |")
    lines.append("|---|---|")

    for name, cmd in GATES:
        rc, _ = _run(cmd)
        status = "PASS" if rc == 0 else "FAIL"
        lines.append(f"| {name} | {status} |")

    lines.append("")
    lines.append("## Open deprecations")
    lines.append("- None tracked in repository metadata.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

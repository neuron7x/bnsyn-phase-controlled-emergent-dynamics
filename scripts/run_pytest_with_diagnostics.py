#!/usr/bin/env python3
"""Run pytest and always emit failure diagnostics while preserving pytest exit code."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pytest with post-run diagnostics")
    parser.add_argument("--markers", default="not (validation or property)")
    parser.add_argument("--junit", type=Path, default=Path("artifacts/tests/junit-fast.xml"))
    parser.add_argument("--log", type=Path, default=Path("artifacts/tests/pytest-fast.log"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/tests/failure-diagnostics.json"))
    parser.add_argument("--output-md", type=Path, default=Path("artifacts/tests/failure-diagnostics.md"))
    parser.add_argument("--schema", type=Path, default=Path("schemas/pytest-failure-diagnostics.schema.json"))
    parser.add_argument("--annotations-file", type=Path, default=None)
    parser.add_argument("--max-annotations", type=int, default=10)
    parser.add_argument("--write-step-summary", action="store_true")
    parser.add_argument("pytest_args", nargs="*")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    full_args = ["-q"]
    if args.markers:
        full_args = ["-m", args.markers, *full_args]
    full_args.extend(args.pytest_args)
    summary_path = Path(os.environ["GITHUB_STEP_SUMMARY"]) if args.write_step_summary and os.environ.get("GITHUB_STEP_SUMMARY") else None
    from bnsyn.qa.pytest_failure_diagnostics import PublicationOptions, run_pytest_with_diagnostics

    publication = PublicationOptions(
        annotations_file=args.annotations_file,
        max_annotations=args.max_annotations,
        github_step_summary=summary_path,
    )
    result = run_pytest_with_diagnostics(
        pytest_args=full_args,
        junit_xml=args.junit,
        log_file=args.log,
        output_json=args.output_json,
        output_md=args.output_md,
        schema_path=args.schema,
        publication=publication,
    )
    return result.pytest_exit_code


if __name__ == "__main__":
    raise SystemExit(main())

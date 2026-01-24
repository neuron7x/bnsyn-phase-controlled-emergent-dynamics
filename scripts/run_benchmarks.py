#!/usr/bin/env python3
"""Run deterministic BN-Syn performance benchmarks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.reporting import format_table
from benchmarks.runner import run_suite, write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic BN-Syn benchmarks.")
    parser.add_argument(
        "--suite",
        choices=("micro", "full"),
        default="micro",
        help="Benchmark suite to run.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON output path. Defaults to benchmarks/results/<suite>-<timestamp>.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_suite(args.suite)
    print(format_table(payload))
    if args.json_out:
        output_path = Path(args.json_out)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path("benchmarks/results") / f"{args.suite}-{timestamp}.json"
    write_json_report(payload, output_path)
    print(f"\nWrote JSON report to {output_path}")


if __name__ == "__main__":
    main()

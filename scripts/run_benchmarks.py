#!/usr/bin/env python3
"""Run deterministic BN-Syn benchmarks."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from benchmarks.reporting import render_table
from benchmarks.runner import run_suite, write_json


def default_output_path(suite: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("benchmarks/results") / f"{suite}_{timestamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="BN-Syn deterministic benchmarks")
    parser.add_argument(
        "--suite",
        choices=["micro", "full"],
        default="micro",
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Output JSON path (default: benchmarks/results/<suite>_<timestamp>.json)",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per scenario")
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured runs per scenario (median reported)",
    )

    args = parser.parse_args()

    result = run_suite(args.suite, warmup_runs=args.warmup, measured_runs=args.runs)
    output_path = Path(args.json_out) if args.json_out else default_output_path(args.suite)
    write_json(result, output_path)
    print(render_table(result))
    print(f"\nJSON written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

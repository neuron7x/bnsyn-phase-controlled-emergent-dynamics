"""Analyze telemetry JSONL into run_health_summary.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from bnsyn.observability.telemetry import latest_run_id, write_run_health_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize BN-Syn telemetry event streams")
    parser.add_argument("events", type=Path, help="Path to logs/events.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Destination JSON path (default: ../run_health_summary.json)")
    parser.add_argument("--run-id", default=None, help="Optional run_id to summarize; defaults to latest run in the stream")
    args = parser.parse_args()

    run_id = args.run_id if args.run_id is not None else latest_run_id(args.events)
    output_path = write_run_health_summary(args.events, output_path=args.output, run_id=run_id)
    print(output_path.as_posix())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate mutation baseline schema contract (fail-closed)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from scripts.mutation_counts import load_mutation_baseline

REQUIRED_TOP_LEVEL_KEYS = {
    "version",
    "timestamp",
    "baseline_score",
    "tolerance_delta",
    "status",
    "description",
    "config",
    "scope",
    "metrics",
}


REQUIRED_METRICS_KEYS = {
    "total_mutants",
    "killed_mutants",
    "survived_mutants",
    "timeout_mutants",
    "suspicious_mutants",
    "score_percent",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate mutation baseline schema")
    parser.add_argument(
        "--baseline",
        default="quality/mutation_baseline.json",
        type=Path,
        help="Path to mutation baseline JSON",
    )
    args = parser.parse_args()

    try:
        payload = json.loads(args.baseline.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("baseline payload must be an object")

        missing_top = REQUIRED_TOP_LEVEL_KEYS.difference(payload.keys())
        if missing_top:
            raise KeyError(f"missing top-level keys: {sorted(missing_top)}")

        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be an object")

        missing_metrics = REQUIRED_METRICS_KEYS.difference(metrics.keys())
        if missing_metrics:
            raise KeyError(f"missing metrics keys: {sorted(missing_metrics)}")

        if not isinstance(payload["baseline_score"], (int, float)):
            raise TypeError("baseline_score must be numeric")
        if not isinstance(payload["tolerance_delta"], (int, float)):
            raise TypeError("tolerance_delta must be numeric")
        if not isinstance(payload["status"], str):
            raise TypeError("status must be string")

        load_mutation_baseline(args.baseline)
    except Exception as exc:
        print(f"❌ Invalid mutation baseline: {exc}", file=sys.stderr)
        return 1

    print(f"✅ Mutation baseline valid: {args.baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

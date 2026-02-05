#!/usr/bin/env python3
"""Generate GitHub Actions mutation summary from canonical mutmut result IDs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.mutation_counts import MutationCounts, calculate_score, read_mutation_counts


def load_baseline(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def render_markdown(
    *, counts: MutationCounts, score: float, baseline_score: float, tolerance: float
) -> str:
    min_acceptable = baseline_score - tolerance
    delta = round(score - baseline_score, 2)
    threshold_gap = round(score - min_acceptable, 2)
    status = "✅ PASS" if score >= min_acceptable else "❌ FAIL"

    lines = [
        "## Mutation Testing Results",
        "",
        f"**Gate Status:** {status}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mutation score | {score:.2f}% |",
        f"| Baseline score | {baseline_score:.2f}% |",
        f"| Tolerance delta | ±{tolerance:.2f}% |",
        f"| Minimum acceptable | {min_acceptable:.2f}% |",
        f"| Delta vs baseline | {delta:+.2f}% |",
        f"| Gap vs minimum acceptable | {threshold_gap:+.2f}% |",
        f"| Killed (incl. timeout) | {counts.killed_equivalent} |",
        f"| Survived | {counts.survived} |",
        f"| Timeout | {counts.timeout} |",
        f"| Suspicious | {counts.suspicious} |",
        f"| Scored denominator | {counts.total_scored} |",
        f"| Skipped | {counts.skipped} |",
        f"| Untested | {counts.untested} |",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CI summary for mutation testing")
    parser.add_argument(
        "--baseline",
        default="quality/mutation_baseline.json",
        type=Path,
        help="Path to mutation baseline JSON",
    )
    args = parser.parse_args()

    summary_path_raw = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path_raw:
        print("❌ GITHUB_STEP_SUMMARY is not set.", file=sys.stderr)
        return 1

    baseline = load_baseline(args.baseline)
    counts = read_mutation_counts()
    score = calculate_score(counts)
    baseline_score = float(baseline["baseline_score"])
    tolerance = float(baseline["tolerance_delta"])

    markdown = render_markdown(
        counts=counts,
        score=score,
        baseline_score=baseline_score,
        tolerance=tolerance,
    )

    with Path(summary_path_raw).open("a", encoding="utf-8") as summary_file:
        summary_file.write(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

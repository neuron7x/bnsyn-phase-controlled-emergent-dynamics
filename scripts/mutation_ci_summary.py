#!/usr/bin/env python3
"""Emit canonical mutation CI outputs and GitHub summary."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.mutation_counts import (
    MutationAssessment,
    assess_mutation_gate,
    load_mutation_baseline,
    read_mutation_counts,
    render_ci_summary_markdown,
    render_github_output_lines,
)


def write_github_output(path: Path, assessment: MutationAssessment) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(render_github_output_lines(assessment))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate canonical mutation CI outputs and summary")
    parser.add_argument(
        "--baseline",
        default="quality/mutation_baseline.json",
        type=Path,
        help="Path to mutation baseline JSON",
    )
    parser.add_argument(
        "--write-output",
        action="store_true",
        help="Write canonical metrics to $GITHUB_OUTPUT",
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="Write markdown report to $GITHUB_STEP_SUMMARY",
    )
    args = parser.parse_args()

    if not args.write_output and not args.write_summary:
        print("❌ No output target selected. Use --write-output and/or --write-summary.", file=sys.stderr)
        return 1

    baseline = load_mutation_baseline(args.baseline)
    counts = read_mutation_counts()
    assessment = assess_mutation_gate(counts, baseline)

    if args.write_output:
        output_path_raw = os.environ.get("GITHUB_OUTPUT")
        if not output_path_raw:
            print("❌ GITHUB_OUTPUT is not set.", file=sys.stderr)
            return 1
        write_github_output(Path(output_path_raw), assessment)

    if args.write_summary:
        summary_path_raw = os.environ.get("GITHUB_STEP_SUMMARY")
        if not summary_path_raw:
            print("❌ GITHUB_STEP_SUMMARY is not set.", file=sys.stderr)
            return 1
        markdown = render_ci_summary_markdown(assessment)
        with Path(summary_path_raw).open("a", encoding="utf-8") as summary_file:
            summary_file.write(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

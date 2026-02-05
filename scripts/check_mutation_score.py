#!/usr/bin/env python3
"""Check mutation score against baseline with tolerance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.mutation_counts import MutationCounts, calculate_score, read_mutation_counts


def load_baseline() -> dict:
    """Load baseline from JSON file."""
    baseline_path = Path("quality/mutation_baseline.json")

    if not baseline_path.exists():
        print(f"❌ Error: Baseline file not found: {baseline_path}", file=sys.stderr)
        print("   Run 'python scripts/generate_mutation_baseline.py' first.", file=sys.stderr)
        sys.exit(1)

    with baseline_path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_mutmut_results() -> MutationCounts:
    """Read canonical mutmut counts from mutmut result-ids output."""
    try:
        return read_mutation_counts()
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running mutmut result-ids: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    """Check mutation score against baseline."""
    parser = argparse.ArgumentParser(description="Check mutation score against baseline")
    parser.add_argument(
        "--strict", action="store_true", help="Fail if baseline is uninitialized (for CI/nightly)"
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Warn but don't fail if baseline is uninitialized (for PR checks)",
    )
    parser.add_argument(
        "current_score",
        nargs="?",
        type=float,
        help="Current mutation score (optional, will read mutmut result-ids if not provided)",
    )
    args = parser.parse_args()

    strict_mode = args.strict
    if not args.strict and not args.advisory:
        strict_mode = False

    baseline = load_baseline()

    status = baseline.get("status", "")
    total_mutants = baseline.get("metrics", {}).get("total_mutants", 0)

    if status == "needs_regeneration" or total_mutants == 0:
        print("⚠️  Mutation Baseline Not Initialized")
        print("=" * 60)
        print("The mutation baseline has not been populated with real data yet.")
        print()
        print("To generate the baseline, run:")
        print("  make mutation-baseline")
        print()
        print("This will take approximately 30 minutes.")
        print()

        if strict_mode:
            print("❌ FAIL: Baseline is uninitialized (strict mode)")
            print("   Nightly/scheduled runs MUST have a valid baseline.")
            return 1

        print("Skipping mutation score check (advisory mode, not blocking).")
        return 0

    baseline_score = baseline["baseline_score"]
    if baseline_score == 0.0 and total_mutants > 0:
        print("⚠️  Baseline score is 0.0 with non-zero mutants; baseline may be stale.")
        print("   Regenerate baseline with canonical result-ids counts: make mutation-baseline")
        if strict_mode:
            print("❌ FAIL: Invalid baseline in strict mode")
            return 1

    tolerance = baseline["tolerance_delta"]
    min_acceptable = baseline_score - tolerance

    print("📊 Mutation Score Check")
    print("=" * 60)
    print(f"Baseline score:     {baseline_score}%")
    print(f"Tolerance:          ±{tolerance}%")
    print(f"Min acceptable:     {min_acceptable}%")
    print()

    if args.current_score is not None:
        current_score = args.current_score
        counts = MutationCounts(0, 0, 0, 0, 0, 0)
    else:
        print("Reading current mutation results...")
        counts = parse_mutmut_results()
        current_score = calculate_score(counts)
        print(f"Total mutants:      {counts.total_scored}")
        print(f"Killed mutants:     {counts.killed_equivalent}")

    print(f"Current score:      {current_score}%")
    print()

    if current_score >= min_acceptable:
        print(f"✅ PASS: Score {current_score}% meets threshold {min_acceptable}%")

        delta = current_score - baseline_score
        if delta > 0:
            print(f"   (+{delta:.2f}% improvement from baseline)")
        elif delta < 0:
            print(f"   ({delta:.2f}% from baseline, within tolerance)")
        else:
            print("   (matches baseline)")

        return 0

    print(f"❌ FAIL: Score {current_score}% below threshold {min_acceptable}%")
    shortfall = min_acceptable - current_score
    print(f"   (Shortfall: {shortfall:.2f}%)")
    print()
    print("Action required:")
    print("  1. Review surviving mutants: mutmut show --status survived")
    print("  2. Add tests to kill surviving mutants")
    print("  3. Re-run mutation testing")
    print("  4. If baseline is outdated, regenerate it:")
    print("     python scripts/generate_mutation_baseline.py")

    return 1


if __name__ == "__main__":
    sys.exit(main())

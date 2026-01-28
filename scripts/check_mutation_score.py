#!/usr/bin/env python3
"""Check mutation score against baseline with tolerance.

This script:
1. Loads the baseline from quality/mutation_baseline.json
2. Reads current mutation score from mutmut results
3. Compares with tolerance and exits non-zero if score is too low

Usage:
    python scripts/check_mutation_score.py [current_score]
    
If current_score is not provided, it will try to run 'mutmut results' to get it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def load_baseline() -> dict:
    """Load baseline from JSON file."""
    baseline_path = Path("quality/mutation_baseline.json")
    
    if not baseline_path.exists():
        print(f"❌ Error: Baseline file not found: {baseline_path}", file=sys.stderr)
        print("   Run 'python scripts/generate_mutation_baseline.py' first.", file=sys.stderr)
        sys.exit(1)
    
    with baseline_path.open() as f:
        return json.load(f)


def parse_mutmut_results() -> tuple[float, int, int]:
    """Parse mutmut results to get current score.
    
    Returns:
        Tuple of (score, total_mutants, killed_mutants)
    """
    try:
        result = subprocess.run(
            ["mutmut", "results"],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running mutmut results: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse counts from output
    counts = {"killed": 0, "survived": 0, "timeout": 0, "suspicious": 0}
    
    for line in output.splitlines():
        line = line.strip().lower()
        if line.startswith("killed:"):
            counts["killed"] = int(line.split(":")[1].strip())
        elif line.startswith("survived:"):
            counts["survived"] = int(line.split(":")[1].strip())
        elif line.startswith("timeout:"):
            counts["timeout"] = int(line.split(":")[1].strip())
        elif line.startswith("suspicious:"):
            counts["suspicious"] = int(line.split(":")[1].strip())
    
    total = counts["killed"] + counts["survived"] + counts["timeout"] + counts["suspicious"]
    if total == 0:
        return 0.0, 0, 0
    
    killed_equivalent = counts["killed"] + counts["timeout"]
    score = round(100.0 * killed_equivalent / total, 2)
    
    return score, total, killed_equivalent


def main() -> int:
    """Check mutation score against baseline."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Check mutation score against baseline")
    parser.add_argument("--strict", action="store_true", 
                        help="Fail if baseline is uninitialized (for CI/nightly)")
    parser.add_argument("--advisory", action="store_true", 
                        help="Warn but don't fail if baseline is uninitialized (for PR checks)")
    parser.add_argument("current_score", nargs="?", type=float, 
                        help="Current mutation score (optional, will run mutmut if not provided)")
    args = parser.parse_args()
    
    # Default to advisory mode if neither specified
    strict_mode = args.strict
    if not args.strict and not args.advisory:
        strict_mode = False  # Default to advisory
    
    # Load baseline
    baseline = load_baseline()
    
    # Check if baseline needs regeneration
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
        else:
            print("Skipping mutation score check (advisory mode, not blocking).")
            return 0
    
    baseline_score = baseline["baseline_score"]
    tolerance = baseline["tolerance_delta"]
    min_acceptable = baseline_score - tolerance
    
    print("📊 Mutation Score Check")
    print("=" * 60)
    print(f"Baseline score:     {baseline_score}%")
    print(f"Tolerance:          ±{tolerance}%")
    print(f"Min acceptable:     {min_acceptable}%")
    print()
    
    # Get current score
    if args.current_score is not None:
        # Score provided as argument
        current_score = args.current_score
        total = 0
        killed = 0
    else:
        # Get score from mutmut results
        print("Reading current mutation results...")
        current_score, total, killed = parse_mutmut_results()
        print(f"Total mutants:      {total}")
        print(f"Killed mutants:     {killed}")
    
    print(f"Current score:      {current_score}%")
    print()
    
    # Compare
    if current_score >= min_acceptable:
        print(f"✅ PASS: Score {current_score}% meets threshold {min_acceptable}%")
        
        # Show delta
        delta = current_score - baseline_score
        if delta > 0:
            print(f"   (+{delta:.2f}% improvement from baseline)")
        elif delta < 0:
            print(f"   ({delta:.2f}% from baseline, within tolerance)")
        else:
            print(f"   (matches baseline)")
        
        return 0
    else:
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

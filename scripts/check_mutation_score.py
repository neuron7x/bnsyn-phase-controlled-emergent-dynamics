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
    # Load baseline
    baseline = load_baseline()
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
    if len(sys.argv) > 1:
        # Score provided as argument
        try:
            current_score = float(sys.argv[1])
            total = 0
            killed = 0
        except ValueError:
            print(f"❌ Error: Invalid score argument: {sys.argv[1]}", file=sys.stderr)
            return 1
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

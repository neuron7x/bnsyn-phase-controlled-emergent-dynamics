#!/usr/bin/env python3
"""Generate mutation testing baseline with real data.

This script:
1. Runs mutmut on specified modules
2. Extracts actual counts and scores
3. Writes a factual baseline to quality/mutation_baseline.json

Usage:
    python scripts/generate_mutation_baseline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import argparse
import re


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_mutmut_version() -> str:
    """Get mutmut version."""
    try:
        result = subprocess.run(
            ["mutmut", "version"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Extract version from output like "mutmut 2.4.5"
        version_line = result.stdout.strip()
        return version_line.split()[-1] if version_line else "unknown"
    except subprocess.CalledProcessError:
        return "unknown"


def parse_mutmut_results(results_output: str) -> dict[str, int]:
    """Parse mutmut results output to extract counts.

    Args:
        results_output: Output from 'mutmut results' command

    Returns:
        Dictionary with counts for each mutation status
    """
    counts = {
        "killed": 0,
        "survived": 0,
        "timeout": 0,
        "suspicious": 0,
        "skipped": 0,
    }

    pattern = re.compile(
        r"^(killed|survived|timeout|suspicious|skipped)[:\s].*?(\d+)\)?$", re.IGNORECASE
    )

    for line in results_output.splitlines():
        line = line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(("killed:", "survived:", "timeout:", "suspicious:", "skipped:")):
            key, value = lowered.split(":", 1)
            counts[key] = int(value.strip())
            continue
        match = pattern.match(line)
        if match:
            key = match.group(1).lower()
            counts[key] = int(match.group(2))

    return counts


def calculate_score(counts: dict[str, int]) -> float:
    """Calculate mutation score percentage.

    Score = (killed + timeout) / (killed + survived + timeout + suspicious) * 100

    Args:
        counts: Dictionary with mutation status counts

    Returns:
        Mutation score as percentage (0-100)
    """
    total = counts["killed"] + counts["survived"] + counts["timeout"] + counts["suspicious"]
    if total == 0:
        return 0.0

    killed_equivalent = counts["killed"] + counts["timeout"]
    return round(100.0 * killed_equivalent / total, 2)


def main() -> int:
    """Generate mutation baseline."""
    parser = argparse.ArgumentParser(description="Generate mutation testing baseline.")
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse existing mutmut cache/results without re-running mutmut.",
    )
    args = parser.parse_args()

    print("🧬 Generating mutation testing baseline...")
    print()

    # Define modules to mutate
    modules = [
        "src/bnsyn/neuron/adex.py",
        "src/bnsyn/plasticity/stdp.py",
        "src/bnsyn/plasticity/three_factor.py",
        "src/bnsyn/temperature/schedule.py",
    ]

    cache_file = Path(".mutmut-cache")
    if not args.reuse_cache:
        if cache_file.exists():
            print(f"Removing existing cache: {cache_file}")
            cache_file.unlink()

        # Run mutmut
        print("Running mutmut (this may take several minutes)...")
        paths_to_mutate = ",".join(modules)

        try:
            run_result = subprocess.run(
                [
                    "mutmut",
                    "run",
                    f"--paths-to-mutate={paths_to_mutate}",
                    "--tests-dir=tests",
                    "--runner=pytest -x -q -m 'not validation and not property and not benchmark'",
                ],
                check=False,  # mutmut returns non-zero when mutants survive
                capture_output=True,
                text=True,
            )
        except Exception as e:
            print(f"Error running mutmut: {e}", file=sys.stderr)
            return 1
        run_output = f"{run_result.stdout}\n{run_result.stderr}".strip()
        if "RuntimeError: Tests don't run cleanly without mutations" in run_output:
            print("❌ Mutmut failed because the test suite did not run cleanly.", file=sys.stderr)
            print(run_output, file=sys.stderr)
            return 1
    elif not cache_file.exists():
        print(
            "❌ Mutmut cache not found; run without --reuse-cache to generate results.",
            file=sys.stderr,
        )
        return 1

    # Get results
    print("Extracting results...")
    try:
        result = subprocess.run(
            ["mutmut", "results"],
            capture_output=True,
            text=True,
            check=True,
        )
        results_output = result.stdout
        print(results_output)
    except subprocess.CalledProcessError as e:
        print(f"Error getting mutmut results: {e}", file=sys.stderr)
        return 1

    # Parse results
    if not re.search(
        r"(killed|survived|timeout|suspicious|skipped)[\s:]", results_output, re.IGNORECASE
    ):
        print("❌ Mutmut results returned no mutation counts.", file=sys.stderr)
        print(results_output, file=sys.stderr)
        return 1
    counts = parse_mutmut_results(results_output)
    score = calculate_score(counts)
    total_mutants = counts["killed"] + counts["survived"] + counts["timeout"] + counts["suspicious"]

    print()
    print(f"Total mutants: {total_mutants}")
    print(f"Killed: {counts['killed']}")
    print(f"Survived: {counts['survived']}")
    print(f"Timeout: {counts['timeout']}")
    print(f"Mutation score: {score}%")
    print()

    # Build baseline JSON
    # Determine status based on whether we have real data
    if total_mutants > 0:
        status = "active"
    else:
        status = "needs_regeneration"

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    baseline = {
        "version": "1.0.0",
        "timestamp": timestamp,
        "baseline_score": score,
        "tolerance_delta": 5.0,
        "status": status,
        "description": "Mutation testing baseline for BNsyn critical modules",
        "config": {
            "tool": "mutmut",
            "tool_version": get_mutmut_version(),
            "python_version": get_python_version(),
            "commit_sha": get_git_commit(),
            "test_command": "pytest -x -q -m 'not validation and not property and not benchmark'",
            "mutation_timeout": 10,
        },
        "scope": {
            "modules": modules,
            "test_markers": "not validation and not property and not benchmark",
        },
        "metrics": {
            "total_mutants": total_mutants,
            "killed_mutants": counts["killed"],
            "survived_mutants": counts["survived"],
            "timeout_mutants": counts["timeout"],
            "suspicious_mutants": counts["suspicious"],
            "score_percent": score,
        },
        "metrics_per_module": {
            module: {"note": "Per-module breakdown requires manual mutmut analysis"}
            for module in modules
        },
        "history": [
            {
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "score": score,
                "commit": get_git_commit()[:8],
                "comment": "Baseline generated by scripts/generate_mutation_baseline.py",
            }
        ],
    }

    # Write to file
    output_path = Path("quality/mutation_baseline.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(baseline, f, indent=2)

    print(f"✅ Baseline written to: {output_path}")
    print()
    print("Baseline summary:")
    print(f"  - Score: {score}%")
    print(f"  - Total mutants: {total_mutants}")
    print(f"  - Commit: {get_git_commit()[:8]}")
    print(f"  - Mutmut version: {get_mutmut_version()}")
    print(f"  - Python version: {get_python_version()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Self-contained coverage validator.

This script validates coverage.json against a threshold with no external dependencies.
It serves as the PRIMARY quality gate for coverage, making Codecov an observability layer only.

Exit codes:
    0: Coverage meets or exceeds threshold
    1: Coverage below threshold or validation error
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# SSOT: Coverage threshold from pyproject.toml and Makefile
DEFAULT_THRESHOLD = 85.0
COVERAGE_FILE = Path("coverage.json")


def load_coverage_data() -> dict[str, Any]:
    """Load and parse coverage.json file."""
    if not COVERAGE_FILE.exists():
        print(f"❌ Error: {COVERAGE_FILE} not found", file=sys.stderr)
        print("   Run: pytest --cov=src/bnsyn --cov-report=json", file=sys.stderr)
        sys.exit(1)

    try:
        with open(COVERAGE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {COVERAGE_FILE}: {e}", file=sys.stderr)
        sys.exit(1)


def calculate_coverage(data: dict[str, Any]) -> tuple[float, int, int]:
    """Calculate total coverage percentage and line counts.
    
    Returns:
        (coverage_percent, covered_lines, total_lines)
    """
    totals = data.get("totals", {})
    
    # Get covered and total lines
    covered_lines = totals.get("covered_lines", 0)
    total_lines = totals.get("num_statements", 0)
    
    if total_lines == 0:
        print("⚠️  Warning: No statements found in coverage report", file=sys.stderr)
        return 0.0, 0, 0
    
    # Calculate percentage
    percent = totals.get("percent_covered", 0.0)
    
    return percent, covered_lines, total_lines


def find_coverage_hotspots(data: dict[str, Any], limit: int = 5) -> list[tuple[str, float]]:
    """Find files with lowest coverage (hotspots for improvement).
    
    Args:
        data: Coverage data dictionary
        limit: Number of hotspots to return
        
    Returns:
        List of (filepath, coverage_percent) tuples, sorted by coverage ascending
    """
    files = data.get("files", {})
    
    file_coverage = []
    for filepath, file_data in files.items():
        summary = file_data.get("summary", {})
        percent = summary.get("percent_covered", 0.0)
        # Only include files with actual statements
        num_statements = summary.get("num_statements", 0)
        if num_statements > 0:
            file_coverage.append((filepath, percent))
    
    # Sort by coverage ascending (lowest first)
    file_coverage.sort(key=lambda x: x[1])
    
    return file_coverage[:limit]


def generate_markdown_report(
    coverage_percent: float,
    covered_lines: int,
    total_lines: int,
    threshold: float,
    hotspots: list[tuple[str, float]],
) -> str:
    """Generate markdown coverage report for GitHub PR comments.
    
    Args:
        coverage_percent: Overall coverage percentage
        covered_lines: Number of covered lines
        total_lines: Total number of statements
        threshold: Coverage threshold
        hotspots: List of (filepath, coverage) tuples for lowest coverage files
        
    Returns:
        Markdown formatted report
    """
    passed = coverage_percent >= threshold
    status_emoji = "✅" if passed else "❌"
    
    report = f"""## {status_emoji} Coverage Report

**Overall Coverage:** {coverage_percent:.2f}% ({covered_lines}/{total_lines} lines)
**Threshold:** {threshold:.1f}%
**Status:** {"PASS" if passed else "FAIL"}

"""
    
    if hotspots:
        report += "### 🔥 Coverage Hotspots (Lowest 5 Files)\n\n"
        report += "| File | Coverage |\n"
        report += "|------|----------|\n"
        for filepath, percent in hotspots:
            # Shorten path for readability
            short_path = filepath.replace("src/bnsyn/", "")
            report += f"| `{short_path}` | {percent:.1f}% |\n"
        report += "\n"
    
    if not passed:
        diff = threshold - coverage_percent
        report += f"**❌ Coverage is {diff:.2f}% below threshold. Please add tests.**\n"
    
    return report


def main() -> None:
    """Main validation logic."""
    # Load coverage data
    print(f"🔍 Loading coverage data from {COVERAGE_FILE}...")
    data = load_coverage_data()
    
    # Calculate coverage
    coverage_percent, covered_lines, total_lines = calculate_coverage(data)
    print(f"📊 Coverage: {coverage_percent:.2f}% ({covered_lines}/{total_lines} lines)")
    
    # Find hotspots
    hotspots = find_coverage_hotspots(data, limit=5)
    
    # Generate report
    threshold = DEFAULT_THRESHOLD
    report = generate_markdown_report(
        coverage_percent, covered_lines, total_lines, threshold, hotspots
    )
    
    # Print report
    print("\n" + report)
    
    # Exit with appropriate code
    if coverage_percent >= threshold:
        print(f"✅ Coverage {coverage_percent:.2f}% ≥ {threshold:.1f}% threshold")
        sys.exit(0)
    else:
        print(f"❌ Coverage {coverage_percent:.2f}% < {threshold:.1f}% threshold", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

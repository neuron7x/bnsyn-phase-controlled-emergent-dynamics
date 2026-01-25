#!/usr/bin/env python3
"""Coverage history tracker - self-hosted coverage trend tracking.

This script maintains a git-based audit trail of coverage metrics over time,
providing observability without external dependencies.

Storage: .coverage-history/YYYY-MM-DD-{sha}.json
Format: {"timestamp": "...", "sha": "...", "coverage": 86.25, "covered": 778, "total": 902}
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

COVERAGE_FILE = Path("coverage.json")
HISTORY_DIR = Path(".coverage-history")


def load_coverage_data() -> dict[str, Any]:
    """Load coverage.json file."""
    if not COVERAGE_FILE.exists():
        print(f"❌ Error: {COVERAGE_FILE} not found", file=sys.stderr)
        print("   Run: pytest --cov=src/bnsyn --cov-report=json", file=sys.stderr)
        sys.exit(1)

    with open(COVERAGE_FILE, encoding="utf-8") as f:
        return json.load(f)


def get_git_sha() -> str:
    """Get current Git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"⚠️  Warning: Could not get Git SHA: {e}", file=sys.stderr)
        return "unknown"


def save_coverage_snapshot(
    coverage_percent: float, covered_lines: int, total_lines: int, git_sha: str
) -> Path:
    """Save coverage snapshot to history directory.
    
    Args:
        coverage_percent: Coverage percentage
        covered_lines: Number of covered lines
        total_lines: Total number of statements
        git_sha: Git commit SHA
        
    Returns:
        Path to saved snapshot file
    """
    # Create history directory if it doesn't exist
    HISTORY_DIR.mkdir(exist_ok=True)
    
    # Generate timestamp and filename
    timestamp = datetime.utcnow().isoformat() + "Z"
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    filename = HISTORY_DIR / f"{date_str}-{git_sha}.json"
    
    # Create snapshot
    snapshot = {
        "timestamp": timestamp,
        "sha": git_sha,
        "coverage": round(coverage_percent, 2),
        "covered": covered_lines,
        "total": total_lines,
    }
    
    # Save to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    
    return filename


def load_history() -> list[dict[str, Any]]:
    """Load all coverage history snapshots.
    
    Returns:
        List of snapshot dictionaries, sorted by timestamp
    """
    if not HISTORY_DIR.exists():
        return []
    
    snapshots = []
    for filepath in HISTORY_DIR.glob("*.json"):
        try:
            with open(filepath, encoding="utf-8") as f:
                snapshot = json.load(f)
                snapshots.append(snapshot)
        except Exception as e:
            print(f"⚠️  Warning: Could not load {filepath}: {e}", file=sys.stderr)
    
    # Sort by timestamp
    snapshots.sort(key=lambda x: x.get("timestamp", ""))
    
    return snapshots


def display_history(snapshots: list[dict[str, Any]], limit: int = 10) -> None:
    """Display coverage history in a table format.
    
    Args:
        snapshots: List of coverage snapshots
        limit: Maximum number of recent snapshots to display
    """
    if not snapshots:
        print("No coverage history found. Run this script after generating coverage.json")
        return
    
    # Display most recent snapshots
    recent = snapshots[-limit:]
    
    print(f"\n📈 Coverage History (last {len(recent)} snapshots)\n")
    print("Date       | SHA     | Coverage | Lines")
    print("-----------|---------|----------|------------")
    
    for snapshot in recent:
        timestamp = snapshot.get("timestamp", "")
        date = timestamp.split("T")[0] if "T" in timestamp else "unknown"
        sha = snapshot.get("sha", "unknown")[:7]
        coverage = snapshot.get("coverage", 0.0)
        covered = snapshot.get("covered", 0)
        total = snapshot.get("total", 0)
        
        print(f"{date} | {sha} | {coverage:6.2f}% | {covered}/{total}")
    
    # Calculate trend
    if len(snapshots) >= 2:
        first_coverage = snapshots[0].get("coverage", 0.0)
        last_coverage = snapshots[-1].get("coverage", 0.0)
        trend = last_coverage - first_coverage
        trend_emoji = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
        
        print(f"\n{trend_emoji} Trend: {trend:+.2f}% from first to latest")


def main() -> None:
    """Main entry point."""
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "view":
        # View mode: display history
        snapshots = load_history()
        display_history(snapshots)
        return
    
    # Track mode: save current coverage
    print("📊 Tracking coverage snapshot...")
    
    # Load current coverage
    data = load_coverage_data()
    totals = data.get("totals", {})
    coverage_percent = totals.get("percent_covered", 0.0)
    covered_lines = totals.get("covered_lines", 0)
    total_lines = totals.get("num_statements", 0)
    
    # Get Git SHA
    git_sha = get_git_sha()
    
    # Save snapshot
    filepath = save_coverage_snapshot(coverage_percent, covered_lines, total_lines, git_sha)
    
    print(f"✅ Saved coverage snapshot to {filepath}")
    print(f"   Coverage: {coverage_percent:.2f}% ({covered_lines}/{total_lines} lines)")
    print(f"   SHA: {git_sha}")
    
    # Display recent history
    snapshots = load_history()
    if len(snapshots) > 1:
        print("\n" + "=" * 60)
        display_history(snapshots, limit=5)


if __name__ == "__main__":
    main()

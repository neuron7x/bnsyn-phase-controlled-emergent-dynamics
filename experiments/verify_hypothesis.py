"""Hypothesis verification utilities.

This module verifies experimental results against hypothesis acceptance criteria
defined in docs/HYPOTHESIS.md.

Usage
-----
python -m experiments.verify_hypothesis docs/HYPOTHESIS.md results/temp_ablation_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_condition_results(results_dir: Path, condition: str) -> dict[str, Any]:
    """Load results for a specific condition.

    Parameters
    ----------
    results_dir : Path
        Results directory.
    condition : str
        Condition name.

    Returns
    -------
    dict[str, Any]
        Condition results.

    Raises
    ------
    FileNotFoundError
        If condition results file not found.
    """
    condition_file = results_dir / f"{condition}.json"
    if not condition_file.exists():
        raise FileNotFoundError(f"Condition results not found: {condition_file}")

    with open(condition_file, encoding="utf-8") as f:
        return json.load(f)


def verify_hypothesis_h1(results_dir: Path) -> tuple[bool, dict[str, Any]]:
    """Verify Hypothesis H1: Temperature-controlled consolidation stability.

    Parameters
    ----------
    results_dir : Path
        Results directory containing condition JSON files.

    Returns
    -------
    bool
        True if hypothesis is supported, False otherwise.
    dict[str, Any]
        Detailed verification results.
    """
    # Load results for cooling and fixed_high conditions
    cooling_results = load_condition_results(results_dir, "cooling_geometric")
    fixed_high_results = load_condition_results(results_dir, "fixed_high")

    cooling_agg = cooling_results["aggregates"]
    fixed_high_agg = fixed_high_results["aggregates"]

    # Extract stability metrics
    cooling_w_cons_var = cooling_agg["stability_w_cons_var_end"]
    fixed_high_w_cons_var = fixed_high_agg["stability_w_cons_var_end"]

    cooling_w_total_var = cooling_agg["stability_w_total_var_end"]
    fixed_high_w_total_var = fixed_high_agg["stability_w_total_var_end"]

    # Compute relative reductions
    if fixed_high_w_cons_var > 0:
        w_cons_reduction = (
            (fixed_high_w_cons_var - cooling_w_cons_var) / fixed_high_w_cons_var
        ) * 100
    else:
        w_cons_reduction = 0.0

    if fixed_high_w_total_var > 0:
        w_total_reduction = (
            (fixed_high_w_total_var - cooling_w_total_var) / fixed_high_w_total_var
        ) * 100
    else:
        w_total_reduction = 0.0

    # Acceptance criterion: at least 10% reduction
    w_cons_pass = w_cons_reduction >= 10.0
    w_total_pass = w_total_reduction >= 10.0
    h1_supported = w_cons_pass and w_total_pass

    verification = {
        "hypothesis": "H1",
        "supported": h1_supported,
        "cooling_w_cons_var": cooling_w_cons_var,
        "fixed_high_w_cons_var": fixed_high_w_cons_var,
        "w_cons_reduction_pct": w_cons_reduction,
        "w_cons_pass": w_cons_pass,
        "cooling_w_total_var": cooling_w_total_var,
        "fixed_high_w_total_var": fixed_high_w_total_var,
        "w_total_reduction_pct": w_total_reduction,
        "w_total_pass": w_total_pass,
    }

    return h1_supported, verification


def main() -> int:
    """Main CLI entry point.

    Returns
    -------
    int
        Exit code (0 = hypothesis supported, 1 = hypothesis refuted, 2 = error).
    """
    parser = argparse.ArgumentParser(
        description="Verify experimental results against hypothesis",
    )
    parser.add_argument(
        "hypothesis_file",
        type=str,
        help="Path to hypothesis document (docs/HYPOTHESIS.md)",
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to results directory (e.g., results/temp_ablation_v1)",
    )

    args = parser.parse_args()

    hypothesis_path = Path(args.hypothesis_file)
    results_path = Path(args.results_dir)

    if not hypothesis_path.exists():
        print(f"Error: Hypothesis file not found: {hypothesis_path}", file=sys.stderr)
        return 2

    if not results_path.exists():
        print(f"Error: Results directory not found: {results_path}", file=sys.stderr)
        return 2

    try:
        supported, verification = verify_hypothesis_h1(results_path)

        print("=" * 60)
        print("Hypothesis Verification: H1")
        print("=" * 60)
        print(f"cooling_geometric w_cons variance: {verification['cooling_w_cons_var']:.6f}")
        print(f"fixed_high w_cons variance:       {verification['fixed_high_w_cons_var']:.6f}")
        print(f"w_cons reduction:                 {verification['w_cons_reduction_pct']:.2f}%")
        print(f"w_cons criterion (≥10%):          {'PASS' if verification['w_cons_pass'] else 'FAIL'}")
        print()
        print(f"cooling_geometric w_total variance: {verification['cooling_w_total_var']:.6f}")
        print(f"fixed_high w_total variance:       {verification['fixed_high_w_total_var']:.6f}")
        print(f"w_total reduction:                 {verification['w_total_reduction_pct']:.2f}%")
        print(f"w_total criterion (≥10%):          {'PASS' if verification['w_total_pass'] else 'FAIL'}")
        print()
        print("=" * 60)
        print(f"H1 VERDICT: {'SUPPORTED' if supported else 'REFUTED'}")
        print("=" * 60)

        return 0 if supported else 1

    except Exception as e:
        print(f"Error during verification: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

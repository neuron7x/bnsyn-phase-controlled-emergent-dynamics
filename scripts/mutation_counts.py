#!/usr/bin/env python3
"""Canonical mutmut count extraction for mutation scoring."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess


@dataclass(frozen=True)
class MutationCounts:
    """Normalized mutmut result counts for scoring."""

    killed: int
    survived: int
    timeout: int
    suspicious: int
    skipped: int
    untested: int

    @property
    def total_scored(self) -> int:
        """Total mutants included in score denominator."""
        return self.killed + self.survived + self.timeout + self.suspicious

    @property
    def killed_equivalent(self) -> int:
        """Mutants counted as killed for score numerator."""
        return self.killed + self.timeout


def _count_ids_for_status(status: str) -> int:
    """Count mutmut IDs for a given status using machine-stable result-ids output."""
    result = subprocess.run(
        ["mutmut", "result-ids", status],
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.strip()
    if not output:
        return 0
    return len(output.split())


def read_mutation_counts() -> MutationCounts:
    """Read canonical mutation counts from mutmut result IDs."""
    return MutationCounts(
        killed=_count_ids_for_status("killed"),
        survived=_count_ids_for_status("survived"),
        timeout=_count_ids_for_status("timeout"),
        suspicious=_count_ids_for_status("suspicious"),
        skipped=_count_ids_for_status("skipped"),
        untested=_count_ids_for_status("untested"),
    )


def calculate_score(counts: MutationCounts) -> float:
    """Calculate mutation score as percentage."""
    if counts.total_scored == 0:
        return 0.0
    return round(100.0 * counts.killed_equivalent / counts.total_scored, 2)

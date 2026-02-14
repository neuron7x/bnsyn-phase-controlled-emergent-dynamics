from __future__ import annotations

import json
from pathlib import Path

from scripts import release_readiness


def test_check_mutation_baseline_rejects_trivial_baseline(tmp_path: Path) -> None:
    baseline_path = tmp_path / "mutation_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "status": "active",
                "metrics": {
                    "total_mutants": 10,
                    "killed_mutants": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    result = release_readiness.check_mutation_baseline(baseline_path)

    assert result.status == "fail"
    assert result.blocking is True
    assert "killed_mutants" in result.details


def test_check_entropy_gate_passes_when_metrics_match(tmp_path: Path) -> None:
    entropy_dir = tmp_path / "entropy"
    entropy_dir.mkdir(parents=True)
    (entropy_dir / "policy.json").write_text(
        json.dumps(
            {
                "comparators": {
                    "process.required_checks_files_present": "eq",
                    "process.gh_workflows_missing_job_timeout_count": "lte",
                }
            }
        ),
        encoding="utf-8",
    )
    (entropy_dir / "baseline.json").write_text(
        json.dumps(
            {
                "process": {
                    "required_checks_files_present": True,
                    "gh_workflows_missing_job_timeout_count": 5,
                }
            }
        ),
        encoding="utf-8",
    )

    original = release_readiness.compute_metrics
    try:
        release_readiness.compute_metrics = lambda _repo_root: {
            "process": {
                "required_checks_files_present": True,
                "gh_workflows_missing_job_timeout_count": 4,
            }
        }
        result = release_readiness.check_entropy_gate(tmp_path)
    finally:
        release_readiness.compute_metrics = original

    assert result.status == "pass"
    assert result.blocking is True

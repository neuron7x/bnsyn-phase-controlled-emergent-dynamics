"""Unit tests for mutation testing scripts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.mutation_counts import MutationCounts, calculate_score


def test_mutation_baseline_structure() -> None:
    """Test that mutation baseline has required structure."""
    baseline_path = Path("quality/mutation_baseline.json")

    assert baseline_path.exists(), "Mutation baseline file must exist"

    with baseline_path.open() as f:
        baseline = json.load(f)

    required_keys = {
        "version",
        "timestamp",
        "baseline_score",
        "tolerance_delta",
        "description",
        "config",
        "scope",
        "metrics",
    }
    assert required_keys.issubset(baseline.keys()), (
        f"Missing keys: {required_keys - baseline.keys()}"
    )

    config_keys = {"tool", "tool_version", "python_version", "commit_sha", "test_command"}
    assert config_keys.issubset(baseline["config"].keys()), (
        f"Missing config keys: {config_keys - baseline['config'].keys()}"
    )

    metrics_keys = {"total_mutants", "killed_mutants", "survived_mutants", "score_percent"}
    assert metrics_keys.issubset(baseline["metrics"].keys()), (
        f"Missing metrics keys: {metrics_keys - baseline['metrics'].keys()}"
    )

    assert isinstance(baseline["metrics"]["total_mutants"], int)
    assert isinstance(baseline["metrics"]["killed_mutants"], int)
    assert isinstance(baseline["metrics"]["survived_mutants"], int)
    assert isinstance(baseline["metrics"]["score_percent"], (int, float))


def test_calculate_mutation_score() -> None:
    """Test mutation score calculation."""
    counts = MutationCounts(
        killed=45,
        survived=5,
        timeout=2,
        suspicious=1,
        skipped=0,
        untested=0,
    )

    score = calculate_score(counts)
    assert isinstance(score, float)
    assert 88.0 < score < 89.0

    counts_zero = MutationCounts(0, 0, 0, 0, 0, 0)
    score_zero = calculate_score(counts_zero)
    assert score_zero == 0.0

    counts_perfect = MutationCounts(100, 0, 0, 0, 0, 0)
    score_perfect = calculate_score(counts_perfect)
    assert score_perfect == 100.0


def test_check_mutation_score_logic() -> None:
    """Test mutation score checking logic."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        baseline = {
            "baseline_score": 75.0,
            "tolerance_delta": 5.0,
        }
        json.dump(baseline, f)
        baseline_path = Path(f.name)

    try:
        baseline_score = 75.0
        tolerance = 5.0
        min_acceptable = baseline_score - tolerance

        assert 72.0 >= min_acceptable
        assert 68.0 < min_acceptable
        assert 70.0 >= min_acceptable
    finally:
        baseline_path.unlink()


def test_mutation_baseline_factuality() -> None:
    """Test that mutation baseline contains factual data (not zeroed defaults)."""
    baseline_path = Path("quality/mutation_baseline.json")

    assert baseline_path.exists(), "Mutation baseline must exist"

    with baseline_path.open() as f:
        baseline = json.load(f)

    metrics = baseline["metrics"]

    has_data = (
        metrics["total_mutants"] > 0
        or metrics["killed_mutants"] > 0
        or metrics["survived_mutants"] > 0
    )

    if not has_data:
        pytest.skip("Baseline not yet generated - run 'make mutation-baseline' first")

    assert metrics["total_mutants"] == (
        metrics["killed_mutants"]
        + metrics["survived_mutants"]
        + metrics.get("timeout_mutants", 0)
        + metrics.get("suspicious_mutants", 0)
    ), "Total mutants must equal sum of categories"

    assert 0.0 <= metrics["score_percent"] <= 100.0, "Score must be between 0 and 100"


def test_mutation_scripts_exist() -> None:
    """Test that mutation scripts exist and are executable."""
    generate_script = Path("scripts/generate_mutation_baseline.py")
    check_script = Path("scripts/check_mutation_score.py")

    assert generate_script.exists(), "Generate baseline script must exist"
    assert check_script.exists(), "Check score script must exist"

    import os

    if os.name != "nt":
        assert os.access(generate_script, os.X_OK), "Generate script should be executable"
        assert os.access(check_script, os.X_OK), "Check script should be executable"


def test_mutation_baseline_version() -> None:
    """Test that baseline has version for tracking changes."""
    baseline_path = Path("quality/mutation_baseline.json")

    with baseline_path.open() as f:
        baseline = json.load(f)

    assert "version" in baseline
    assert isinstance(baseline["version"], str)
    assert "." in baseline["version"], "Version should use semantic versioning (e.g., '1.0.0')"


def test_read_mutation_counts_uses_result_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure canonical count extraction comes from mutmut result-ids output."""
    from types import SimpleNamespace

    from scripts.mutation_counts import read_mutation_counts

    outputs = {
        "killed": "3\n",
        "survived": "1 2\n",
        "timeout": "\n",
        "suspicious": "\n",
        "skipped": "\n",
        "untested": "4 5 6\n",
    }

    def fake_run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        assert args[0:2] == ["mutmut", "result-ids"]
        status = args[2]
        return SimpleNamespace(stdout=outputs[status])

    monkeypatch.setattr("subprocess.run", fake_run)

    counts = read_mutation_counts()

    assert counts == MutationCounts(
        killed=1,
        survived=2,
        timeout=0,
        suspicious=0,
        skipped=0,
        untested=3,
    )
    assert counts.total_scored == 3
    assert counts.killed_equivalent == 1


def test_check_script_reads_nonzero_score_from_result_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure check script score is non-zero when killed-equivalent mutants exist."""
    from types import SimpleNamespace

    import scripts.check_mutation_score as check_mutation_score

    outputs = {
        "killed": "1 2 3\n",
        "survived": "4 5\n",
        "timeout": "\n",
        "suspicious": "\n",
        "skipped": "\n",
        "untested": "6 7 8\n",
    }

    def fake_run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        assert args[0:2] == ["mutmut", "result-ids"]
        return SimpleNamespace(stdout=outputs[args[2]])

    monkeypatch.setattr("subprocess.run", fake_run)

    score, total, killed = check_mutation_score.parse_mutmut_results()
    assert score == 60.0
    assert total == 5
    assert killed == 3

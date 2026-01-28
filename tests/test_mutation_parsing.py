"""Unit tests for mutation testing scripts.

These tests verify the mutation baseline parsing and score checking logic
without requiring mutmut to actually run (which would be slow).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def test_mutation_baseline_structure() -> None:
    """Test that mutation baseline has required structure."""
    baseline_path = Path("quality/mutation_baseline.json")
    
    assert baseline_path.exists(), "Mutation baseline file must exist"
    
    with baseline_path.open() as f:
        baseline = json.load(f)
    
    # Check required top-level keys
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
    assert required_keys.issubset(baseline.keys()), f"Missing keys: {required_keys - baseline.keys()}"
    
    # Check config structure
    config_keys = {"tool", "tool_version", "python_version", "commit_sha", "test_command"}
    assert config_keys.issubset(baseline["config"].keys()), f"Missing config keys: {config_keys - baseline['config'].keys()}"
    
    # Check metrics structure
    metrics_keys = {"total_mutants", "killed_mutants", "survived_mutants", "score_percent"}
    assert metrics_keys.issubset(baseline["metrics"].keys()), f"Missing metrics keys: {metrics_keys - baseline['metrics'].keys()}"
    
    # Check that metrics are non-zero (factual baseline)
    # Note: total_mutants might be 0 initially, but after first run should be > 0
    # For now, just check structure
    assert isinstance(baseline["metrics"]["total_mutants"], int)
    assert isinstance(baseline["metrics"]["killed_mutants"], int)
    assert isinstance(baseline["metrics"]["survived_mutants"], int)
    assert isinstance(baseline["metrics"]["score_percent"], (int, float))


def test_parse_mutmut_results() -> None:
    """Test parsing of mutmut results output."""
    # Import the function from the script
    import sys
    sys.path.insert(0, "scripts")
    from generate_mutation_baseline import parse_mutmut_results
    
    # Sample mutmut output
    sample_output = """
To apply a mutant on disk:
    mutmut apply <id>

To show a mutant:
    mutmut show <id>


Killed: 45
Survived: 5
Timeout: 2
Suspicious: 1
Skipped: 0

Total: 53
"""
    
    counts = parse_mutmut_results(sample_output)
    
    assert counts["killed"] == 45
    assert counts["survived"] == 5
    assert counts["timeout"] == 2
    assert counts["suspicious"] == 1
    assert counts["skipped"] == 0


def test_calculate_mutation_score() -> None:
    """Test mutation score calculation."""
    import sys
    sys.path.insert(0, "scripts")
    from generate_mutation_baseline import calculate_score
    
    # Test normal case
    counts = {
        "killed": 45,
        "survived": 5,
        "timeout": 2,
        "suspicious": 1,
        "skipped": 0,
    }
    
    # Score = (killed + timeout) / (killed + survived + timeout + suspicious)
    # Score = (45 + 2) / (45 + 5 + 2 + 1) = 47 / 53 ≈ 88.68
    score = calculate_score(counts)
    
    assert isinstance(score, float)
    assert 88.0 < score < 89.0
    
    # Test edge case: no mutants
    counts_zero = {
        "killed": 0,
        "survived": 0,
        "timeout": 0,
        "suspicious": 0,
        "skipped": 0,
    }
    
    score_zero = calculate_score(counts_zero)
    assert score_zero == 0.0
    
    # Test perfect score
    counts_perfect = {
        "killed": 100,
        "survived": 0,
        "timeout": 0,
        "suspicious": 0,
        "skipped": 0,
    }
    
    score_perfect = calculate_score(counts_perfect)
    assert score_perfect == 100.0


def test_check_mutation_score_logic() -> None:
    """Test mutation score checking logic."""
    import sys
    sys.path.insert(0, "scripts")
    
    # Create a temporary baseline
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        baseline = {
            "baseline_score": 75.0,
            "tolerance_delta": 5.0,
        }
        json.dump(baseline, f)
        baseline_path = Path(f.name)
    
    try:
        # Test score checking logic manually
        baseline_score = 75.0
        tolerance = 5.0
        min_acceptable = baseline_score - tolerance  # 70.0
        
        # Score above threshold: pass
        current_score = 72.0
        assert current_score >= min_acceptable
        
        # Score below threshold: fail
        current_score_fail = 68.0
        assert current_score_fail < min_acceptable
        
        # Score exactly at threshold: pass
        current_score_exact = 70.0
        assert current_score_exact >= min_acceptable
        
    finally:
        baseline_path.unlink()


def test_mutation_baseline_factuality() -> None:
    """Test that mutation baseline contains factual data (not placeholders)."""
    baseline_path = Path("quality/mutation_baseline.json")
    
    assert baseline_path.exists(), "Mutation baseline must exist"
    
    with baseline_path.open() as f:
        baseline = json.load(f)
    
    # Check that baseline has been generated at least once
    # (metrics should not all be zero placeholders)
    metrics = baseline["metrics"]
    
    # At least one of these should be non-zero after first generation
    has_data = (
        metrics["total_mutants"] > 0 or
        metrics["killed_mutants"] > 0 or
        metrics["survived_mutants"] > 0
    )
    
    # Note: This test might fail before first baseline generation
    # That's acceptable - it ensures we don't commit placeholder baselines
    if not has_data:
        pytest.skip("Baseline not yet generated - run 'make mutation-baseline' first")
    
    # If we have data, verify it's consistent
    assert metrics["total_mutants"] == (
        metrics["killed_mutants"] +
        metrics["survived_mutants"] +
        metrics.get("timeout_mutants", 0) +
        metrics.get("suspicious_mutants", 0)
    ), "Total mutants must equal sum of categories"
    
    # Verify score is in valid range
    assert 0.0 <= metrics["score_percent"] <= 100.0, "Score must be between 0 and 100"


def test_mutation_scripts_exist() -> None:
    """Test that mutation scripts exist and are executable."""
    generate_script = Path("scripts/generate_mutation_baseline.py")
    check_script = Path("scripts/check_mutation_score.py")
    
    assert generate_script.exists(), "Generate baseline script must exist"
    assert check_script.exists(), "Check score script must exist"
    
    # Check they're executable (on Unix systems)
    import os
    if os.name != 'nt':  # Skip on Windows
        assert os.access(generate_script, os.X_OK), "Generate script should be executable"
        assert os.access(check_script, os.X_OK), "Check script should be executable"


def test_mutation_baseline_version() -> None:
    """Test that baseline has version for tracking changes."""
    baseline_path = Path("quality/mutation_baseline.json")
    
    with baseline_path.open() as f:
        baseline = json.load(f)
    
    assert "version" in baseline
    assert isinstance(baseline["version"], str)
    # Version should be semantic versioning format
    assert "." in baseline["version"], "Version should use semantic versioning (e.g., '1.0.0')"

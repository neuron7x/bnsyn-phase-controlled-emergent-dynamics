"""Tests for hypothesis verification with v2 results."""

from pathlib import Path

import pytest


def test_verify_hypothesis_v2_bundled_results() -> None:
    """Test that bundled v2 results pass hypothesis verification."""
    from experiments.verify_hypothesis import verify_hypothesis_h1

    results_dir = Path("results/temp_ablation_v2")
    
    if not results_dir.exists():
        pytest.skip("Bundled v2 results not found (expected in CI)")
    
    supported, verification = verify_hypothesis_h1(results_dir)
    
    # H1 should be supported
    assert supported, "Bundled v2 results should support H1"
    
    # Check consolidation gates pass
    assert verification["consolidation_gates_pass"], "Consolidation gates should pass"
    assert verification["cooling_consolidation_nontrivial"], "Cooling should have non-trivial consolidation"
    assert verification["fixed_high_consolidation_nontrivial"], "Fixed_high should have non-trivial consolidation"
    
    # Check stability improvement
    assert verification["w_total_pass"], "w_total reduction should be >= 10%"
    assert verification["w_total_reduction_pct"] >= 10.0


def test_verify_hypothesis_v1_bundled_results() -> None:
    """Test that bundled v1 results still work (for backward compatibility)."""
    from experiments.verify_hypothesis import verify_hypothesis_h1

    results_dir = Path("results/temp_ablation_v1")
    
    if not results_dir.exists():
        pytest.skip("Bundled v1 results not found")
    
    # v1 uses old verification logic (no consolidation gates for v1 condition name)
    # It should still verify successfully
    supported, verification = verify_hypothesis_h1(results_dir)
    
    # The logic will detect cooling_geometric and not apply strict gates
    # Just check it runs without error
    assert "supported" in verification

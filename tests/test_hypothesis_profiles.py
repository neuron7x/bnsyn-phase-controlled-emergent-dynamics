"""Test Hypothesis profile configuration to prevent CI misconfigurations."""

from __future__ import annotations

import os
import subprocess
import sys


def test_hypothesis_profile_env_var_must_be_valid() -> None:
    """Guard test: HYPOTHESIS_PROFILE env var must be a valid profile name.
    
    This test prevents CI from silently misconfiguring Hypothesis by catching
    invalid profile names at test time. If this test fails, the HYPOTHESIS_PROFILE
    environment variable is set to a profile name that doesn't exist in conftest.py.
    """
    profile = os.getenv("HYPOTHESIS_PROFILE")
    
    # If HYPOTHESIS_PROFILE is not set, test passes (will use CI auto-detect or default)
    if not profile:
        return
    
    # Valid profiles defined in tests/conftest.py
    valid_profiles = {"quick", "thorough", "ci"}
    
    if profile not in valid_profiles:
        raise ValueError(
            f"HYPOTHESIS_PROFILE={profile!r} is not a valid profile. "
            f"Valid profiles: {', '.join(sorted(valid_profiles))}. "
            f"Update tests/conftest.py if you need to add a new profile."
        )


def test_hypothesis_profile_loads_without_error() -> None:
    """Test that Hypothesis profile can be loaded without error.
    
    This test runs a minimal pytest command that will trigger conftest.py
    profile loading and fail if there's a profile error.
    """
    profile = os.getenv("HYPOTHESIS_PROFILE")
    
    if not profile:
        # No profile set, nothing to validate
        return
    
    # Run pytest with --collect-only to test conftest.py loading without running tests
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    # Check for Hypothesis profile errors in output
    output = result.stdout + result.stderr
    
    if "hypothesis.errors.InvalidArgument" in output:
        raise AssertionError(
            f"Hypothesis profile error detected:\n{output}"
        )
    
    if "Failed to load profile" in output:
        raise AssertionError(
            f"Failed to load Hypothesis profile:\n{output}"
        )

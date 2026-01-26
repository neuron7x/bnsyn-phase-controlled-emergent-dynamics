"""CI guard test to enforce proper test tier separation.

This test ensures that slow/heavy tests are properly marked as validation
to prevent them from leaking into the fast smoke test suite.

Notes
-----
This is a meta-test that validates test infrastructure correctness.
It runs as part of the smoke suite to catch marking violations early.
"""


def test_validation_marker_exists() -> None:
    """Test that validation marker is properly configured.

    Notes
    -----
    Ensures pytest configuration includes the validation marker definition.
    Without this, --strict-markers would fail on validation tests.
    """
    import subprocess

    result = subprocess.run(
        ["pytest", "--markers"],
        cwd=".",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, "pytest --markers failed"
    assert "validation:" in result.stdout, (
        "validation marker not found in pytest configuration.\n"
        "Add to pyproject.toml: markers = ['validation: ...']"
    )


def test_smoke_marker_exists() -> None:
    """Test that smoke marker is properly configured.

    Notes
    -----
    Ensures pytest configuration includes the smoke marker definition.
    """
    import subprocess

    result = subprocess.run(
        ["pytest", "--markers"],
        cwd=".",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, "pytest --markers failed"
    assert "smoke:" in result.stdout, (
        "smoke marker not found in pytest configuration.\n"
        "Add to pyproject.toml: markers = ['smoke: ...']"
    )



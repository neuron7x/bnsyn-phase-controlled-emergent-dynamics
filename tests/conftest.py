from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from hypothesis import Verbosity, settings

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Register Hypothesis profiles
settings.register_profile(
    "quick",
    max_examples=100,
    deadline=5000,
    print_blob=True,
    derandomize=True,
)
settings.register_profile(
    "thorough",
    max_examples=1000,
    deadline=20000,
    print_blob=True,
    derandomize=True,
)
settings.register_profile(
    "ci-quick",
    max_examples=50,
    deadline=5000,
    verbosity=Verbosity.verbose,
    print_blob=True,
    derandomize=True,
)

# Load profile based on environment
if os.getenv("CI"):
    settings.load_profile("ci-quick")
elif os.getenv("HYPOTHESIS_PROFILE"):
    settings.load_profile(os.getenv("HYPOTHESIS_PROFILE"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    errors: list[str] = []
    for item in items:
        path = Path(str(item.fspath)).resolve()
        in_validation_dir = "tests/validation" in path.as_posix()
        has_validation_marker = item.get_closest_marker("validation") is not None
        if in_validation_dir and not has_validation_marker:
            errors.append(f"Missing @pytest.mark.validation for {item.nodeid}")
        if not in_validation_dir and has_validation_marker:
            errors.append(f"Validation marker used outside tests/validation: {item.nodeid}")
    if errors:
        raise pytest.UsageError("\n".join(errors))

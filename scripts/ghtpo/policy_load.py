"""Load and validate GHTPO policy."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_policy(repo_root: Path) -> dict[str, Any]:
    policy_path = repo_root / ".github" / "ghtpo.yaml"
    data = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid .github/ghtpo.yaml payload")
    return data

"""Required check manifest for GHTPO."""
from __future__ import annotations

import json
from pathlib import Path


def build_manifest(repo_root: Path) -> dict[str, list[str]]:
    return {"required_checks": ["ghtpo-pr"]}


def write_manifest(repo_root: Path) -> Path:
    out = repo_root / "artifacts" / "ghtpo" / "checks" / "REQUIRED_CHECKS_MANIFEST.json"
    out.write_text(json.dumps(build_manifest(repo_root), indent=2) + "\n", encoding="utf-8")
    return out

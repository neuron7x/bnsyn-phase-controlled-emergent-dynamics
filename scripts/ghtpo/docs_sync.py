"""Synchronize evidence index for GHTPO."""
from __future__ import annotations

from pathlib import Path


def write_evidence_index(repo_root: Path) -> Path:
    out = repo_root / "artifacts" / "ghtpo" / "proof" / "EVIDENCE_INDEX.md"
    out.write_text(
        "# GHTPO Evidence Index\n\n"
        "- Policy: artifacts/ghtpo/policy/POLICY_DECL.json\n"
        "- Checks: artifacts/ghtpo/checks/REQUIRED_CHECKS_MANIFEST.json\n"
        "- Quality: artifacts/ghtpo/quality.json\n",
        encoding="utf-8",
    )
    return out

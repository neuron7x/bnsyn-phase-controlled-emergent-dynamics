#!/usr/bin/env python3
"""Scan governed docs for orphan normative statements.

This script reads the authoritative governed paths list from docs/REPO_STRUCTURE.md
and scans those documents for normative language compliance.

Rules:
- Lines containing "[NORMATIVE]" must include a CLM-#### identifier
- Every referenced CLM-#### must exist in claims/claims.yml

Exit codes:
- 0: All checks pass
- 1: Governed paths could not be parsed from REPO_STRUCTURE.md
- 2: Orphan normative statements found ([NORMATIVE] without CLM tag)
- 3: Invalid CLM references (referenced CLM not in claims.yml)
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_STRUCTURE = ROOT / "docs" / "REPO_STRUCTURE.md"
CLAIMS = ROOT / "claims" / "claims.yml"

CLM_RE = re.compile(r"\bCLM-\d{4}\b")
NORMATIVE_TAG_RE = re.compile(r"\[NORMATIVE\]")

# Files that discuss normative conventions without being bound to claims
ALLOWLIST_LABEL_DOCS = {
    "docs/CONSTITUTIONAL_AUDIT.md",
    "docs/NORMATIVE_LABELING.md",
    "docs/SSOT.md",
    "docs/SSOT_RULES.md",
    "docs/GOVERNANCE.md",
    "docs/REPO_STRUCTURE.md",
    "docs/CRITICALITY_CONTROL_VS_MEASUREMENT.md",
    "README_CLAIMS_GATE.md",
}


def load_governed_paths() -> list[str]:
    """Parse governed_paths YAML block from REPO_STRUCTURE.md."""
    if not REPO_STRUCTURE.exists():
        raise SystemExit(f"Missing REPO_STRUCTURE.md: {REPO_STRUCTURE}")

    text = REPO_STRUCTURE.read_text(encoding="utf-8")
    if "```yaml" not in text:
        raise SystemExit("REPO_STRUCTURE.md missing governed_paths YAML block")

    yaml_block = text.split("```yaml", 1)[1].split("```", 1)[0]
    data = yaml.safe_load(yaml_block)

    if not isinstance(data, dict) or "governed_paths" not in data:
        raise SystemExit("REPO_STRUCTURE.md YAML block must contain 'governed_paths'")

    gp = data["governed_paths"]
    paths: list[str] = []
    for category, items in gp.items():
        if isinstance(items, list):
            paths.extend(str(p) for p in items)
    return paths


def load_claim_ids() -> set[str]:
    """Load all claim IDs from claims.yml."""
    if not CLAIMS.exists():
        raise SystemExit(f"Missing claims file: {CLAIMS}")
    data = yaml.safe_load(CLAIMS.read_text(encoding="utf-8"))
    claims = data.get("claims", [])
    return {str(c.get("id", "")) for c in claims if isinstance(c, dict)}


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def main() -> int:
    governed_paths = load_governed_paths()
    claim_ids = load_claim_ids()

    orphans: list[tuple[str, int, str]] = []
    invalid_refs: list[tuple[str, int, str]] = []
    scanned_files = 0

    for gp in governed_paths:
        path = ROOT / gp
        if not path.exists():
            continue
        if not path.suffix == ".md":
            continue

        scanned_files += 1
        rp = rel(path)
        allow_label_only = rp in ALLOWLIST_LABEL_DOCS

        for ln, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            has_normative_tag = NORMATIVE_TAG_RE.search(line) is not None
            clm_ids_in_line = CLM_RE.findall(line)

            # Check for [NORMATIVE] without CLM
            if has_normative_tag and not clm_ids_in_line and not allow_label_only:
                orphans.append((rp, ln, line.strip()[:100]))

            # Validate all CLM references exist
            for cid in clm_ids_in_line:
                if cid not in claim_ids:
                    invalid_refs.append((rp, ln, cid))

    # Report results
    print(f"[governed-docs] Scanned {scanned_files} governed docs")
    print(f"[governed-docs] Orphan [NORMATIVE] tags (missing CLM): {len(orphans)}")
    print(f"[governed-docs] Invalid CLM references: {len(invalid_refs)}")

    if orphans:
        print("\nERROR: [NORMATIVE] lines missing Claim ID:")
        for rp, ln, line in orphans[:20]:
            print(f"  {rp}:{ln}: {line}")
        return 2

    if invalid_refs:
        print("\nERROR: Invalid CLM references (not in claims.yml):")
        for rp, ln, cid in invalid_refs[:20]:
            print(f"  {rp}:{ln}: {cid}")
        return 3

    print("[governed-docs] OK: governed docs have no orphan normative statements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

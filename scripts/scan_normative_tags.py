#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ssot_rules import RULE_IDS, assert_rule_ids_match

INVENTORY = ROOT / "docs" / "INVENTORY.md"
CLAIMS = ROOT / "claims" / "claims.yml"

NORMATIVE_WORD_RE = re.compile(r"\b(must|required|shall)\b", re.IGNORECASE)
CLM_RE = re.compile(r"CLM-\d{4}")


def fail(msg: str) -> None:
    print(f"[normative-scan] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_claim_ids() -> set[str]:
    data = yaml.safe_load(CLAIMS.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        fail("claims.yml must be a YAML mapping")
    claims = data.get("claims")
    if not isinstance(claims, list) or not claims:
        fail("claims.yml missing claims list")
    ids = set()
    for entry in claims:
        if isinstance(entry, dict) and "id" in entry:
            ids.add(str(entry["id"]))
    return ids


def load_governed_docs() -> list[Path]:
    if not INVENTORY.exists():
        fail("docs/INVENTORY.md not found")
    lines = INVENTORY.read_text(encoding="utf-8").splitlines()
    governed: list[Path] = []
    in_section = False
    for line in lines:
        if line.strip() == "## Governed documents":
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break
            match = re.search(r"`([^`]+)`", line)
            if match:
                governed.append(ROOT / match.group(1))
    if not governed:
        fail("No governed documents listed in docs/INVENTORY.md")
    return governed


def main() -> int:
    assert_rule_ids_match(RULE_IDS)

    claim_ids = load_claim_ids()
    governed_docs = load_governed_docs()

    missing_files = [str(p.relative_to(ROOT)) for p in governed_docs if not p.exists()]
    if missing_files:
        fail(f"Governed documents missing: {missing_files}")

    orphan_normative = []
    invalid_claim_refs = []

    for doc in governed_docs:
        rel = doc.relative_to(ROOT)
        for idx, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            has_normative_tag = "[NORMATIVE]" in line
            has_normative_word = bool(NORMATIVE_WORD_RE.search(line))
            if not has_normative_tag and not has_normative_word:
                continue
            clm_ids = CLM_RE.findall(line)
            if not clm_ids:
                orphan_normative.append(f"{rel}:{idx}: {line.strip()}")
                continue
            for clm in clm_ids:
                if clm not in claim_ids:
                    invalid_claim_refs.append(f"{rel}:{idx}: {clm}")

    if orphan_normative:
        fail("Orphan normative statements found:\n" + "\n".join(orphan_normative))
    if invalid_claim_refs:
        fail("Invalid CLM references found:\n" + "\n".join(invalid_claim_refs))

    print("[normative-scan] OK: governed docs have no orphan normative statements.")
    print(f"  Governed docs: {len(governed_docs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

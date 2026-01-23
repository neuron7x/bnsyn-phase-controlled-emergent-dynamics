"""Scan markdown for malformed normative claim tags.

Enforced invariant (pragmatic):
- In *claim-bearing* docs, any occurrence of '[NORMATIVE]' must include a Claim ID 'CLM-####'
  on the same line.
- Any referenced CLM must exist in claims/claims.yml.

Rationale:
Some governance/policy docs discuss the term '[NORMATIVE]' as a label (without binding to a
specific claim). Those files are explicitly allow-listed.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "claims" / "claims.yml"

CLM_RE = re.compile(r"\bCLM-\d{4}\b")
NORM_RE = re.compile(r"\[NORMATIVE\]")

# Files that may mention [NORMATIVE] as a label/convention (not a claim binding).
ALLOWLIST_LABEL_DOCS = {
    "README_CLAIMS_GATE.md",
    "docs/CONSTITUTIONAL_AUDIT.md",
    "docs/NORMATIVE_LABELING.md",
    "docs/SSOT.md",
    "docs/CRITICALITY_CONTROL_VS_MEASUREMENT.md",
}


def load_claim_ids() -> set[str]:
    data = yaml.safe_load(CLAIMS.read_text(encoding="utf-8"))
    claims = data.get("claims", [])
    out: set[str] = set()
    for c in claims:
        cid = c.get("id")
        if isinstance(cid, str):
            out.add(cid)
    return out


def markdown_files() -> list[Path]:
    files: list[Path] = []
    for p in (ROOT / "docs").rglob("*.md"):
        if "docs/appendix" in str(p):
            continue
        files.append(p)
    for p in ROOT.glob("README*.md"):
        files.append(p)
    return sorted(set(files))


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def main() -> int:
    claim_ids = load_claim_ids()
    malformed = []
    missing = []

    for f in markdown_files():
        rf = rel(f)
        allow_label_only = rf in ALLOWLIST_LABEL_DOCS

        for ln, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if NORM_RE.search(line):
                ids = CLM_RE.findall(line)

                if not ids and not allow_label_only:
                    malformed.append((rf, ln, line))
                for cid in ids:
                    if cid not in claim_ids:
                        missing.append((rf, ln, cid))

    if malformed:
        print("ERROR: [NORMATIVE] lines missing Claim ID (outside allowlist):")
        for rf, ln, line in malformed[:50]:
            print(f"  {rf}:{ln}: {line}")
        return 2

    if missing:
        print("ERROR: [NORMATIVE] references missing Claim IDs:")
        for rf, ln, cid in missing[:50]:
            print(f"  {rf}:{ln}: {cid}")
        return 3

    print("OK: normative tag scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

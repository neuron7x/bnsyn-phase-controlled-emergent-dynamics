#!/usr/bin/env python3
"""
Validate BN-Syn bibliography SSOT:
- bnsyn.bib entries include DOI for Tier-A sources
- mapping.yml is well-formed and references existing bibkeys
- sources.lock lines are syntactically valid and SHA256 matches LOCK_STRING
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception as e:
    print("ERROR: PyYAML is required to run this validator. Install with: pip install pyyaml", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "bibliography" / "bnsyn.bib"
LOCK = ROOT / "bibliography" / "sources.lock"
MAP = ROOT / "bibliography" / "mapping.yml"
CLAIMS = ROOT / "claims" / "claims.yml"

DOI_RE = re.compile(r"doi\s*=\s*\{([^}]+)\}", re.IGNORECASE)
KEY_RE = re.compile(r"@\w+\{([^,]+),")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
TIER_RE = re.compile(r"^Tier-(A|S|C)$")

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def parse_bibtex(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    entries: dict[str, dict[str, str]] = {}
    # crude but robust enough: split by @
    chunks = ["@" + c for c in text.split("@") if c.strip()]
    for c in chunks:
        m = KEY_RE.search(c)
        if not m:
            continue
        key = m.group(1).strip()
        doi_m = DOI_RE.search(c)
        doi = doi_m.group(1).strip() if doi_m else ""
        entries[key] = {"doi": doi}
    return entries

def load_mapping(path: Path) -> dict[str, dict[str, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("mapping.yml must be a YAML mapping")
    out: dict[str, dict[str, str]] = {}
    for clm, v in data.items():
        if not isinstance(clm, str) or not clm.startswith("CLM-"):
            raise SystemExit(f"Invalid CLM key: {clm!r}")
        if not isinstance(v, dict):
            raise SystemExit(f"{clm}: value must be mapping with keys bibkey/tier/section")
        for req in ("bibkey", "tier", "section"):
            if req not in v:
                raise SystemExit(f"{clm}: missing required field {req}")
        bibkey = v["bibkey"]
        tier = v["tier"]
        if not isinstance(bibkey, str) or not bibkey:
            raise SystemExit(f"{clm}: bibkey must be non-empty string")
        if not isinstance(tier, str) or not TIER_RE.match(tier):
            raise SystemExit(f"{clm}: tier must be Tier-A|Tier-S|Tier-C")
        out[clm] = {"bibkey": bibkey, "tier": tier, "section": str(v["section"])}
    return out

def load_claims(path: Path) -> set[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("claims.yml must be a YAML mapping")
    claims = data.get("claims")
    if not isinstance(claims, list) or not claims:
        raise SystemExit("claims.yml must contain a non-empty claims list")
    ids: set[str] = set()
    for entry in claims:
        if not isinstance(entry, dict):
            raise SystemExit("claims.yml claim entry must be a mapping")
        cid = entry.get("id")
        if not isinstance(cid, str) or not cid.startswith("CLM-"):
            raise SystemExit(f"claims.yml invalid claim id: {cid!r}")
        if cid in ids:
            raise SystemExit(f"claims.yml duplicate claim id: {cid}")
        ids.add(cid)
    return ids

def parse_lock(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"sources.lock invalid line (missing '='): {line}")
        bibkey, rest = line.split("=", 1)
        parts = rest.split("::")
        # Tier-A has 5 fields after '='; Tier-S also 5 (with NODOI)
        if len(parts) != 5:
            raise SystemExit(f"sources.lock invalid field count for {bibkey}: expected 5 '::' fields, got {len(parts)}")
        doi_or_nodoi, url, f3, f4, sha = parts
        rows.append({
            "bibkey": bibkey.strip(),
            "doi_or_nodoi": doi_or_nodoi.strip(),
            "url": url.strip(),
            "f3": f3.strip(),
            "f4": f4.strip(),
            "sha": sha.strip(),
        })
    return rows

def main() -> int:
    for p in (BIB, LOCK, MAP, CLAIMS):
        if not p.exists():
            print(f"ERROR: missing required file: {p}", file=sys.stderr)
            return 2

    bib = parse_bibtex(BIB)
    mapping = load_mapping(MAP)
    claim_ids = load_claims(CLAIMS)
    lock_rows = parse_lock(LOCK)

    lock_by_key = {r["bibkey"]: r for r in lock_rows}
    missing_lock = set(bib.keys()) - set(lock_by_key.keys())
    # Only require lock for bibkeys present in bnsyn.bib (including Tier-S misc)
    if missing_lock:
        print(f"ERROR: sources.lock missing bibkeys: {sorted(missing_lock)}", file=sys.stderr)
        return 3

    # Validate mapping references bibkeys
    for clm, v in mapping.items():
        bk = v["bibkey"]
        if bk not in bib:
            print(f"ERROR: {clm} references unknown bibkey: {bk}", file=sys.stderr)
            return 4
        tier = v["tier"]
        if tier == "Tier-A":
            doi = bib[bk]["doi"]
            if not doi:
                print(f"ERROR: {clm} is Tier-A but bibkey {bk} has no DOI", file=sys.stderr)
                return 5

    missing_claims = sorted(set(claim_ids) - set(mapping.keys()))
    if missing_claims:
        print(f"ERROR: claims.yml has unmapped claim IDs: {missing_claims}", file=sys.stderr)
        return 8

    # Validate lock hashes
    for bk, entry in lock_by_key.items():
        sha = entry["sha"]
        if not HEX64_RE.match(sha):
            print(f"ERROR: sources.lock {bk} has invalid SHA256 (must be 64 lowercase hex): {sha}", file=sys.stderr)
            return 6
        doi_or_nodoi = entry["doi_or_nodoi"]
        if doi_or_nodoi == "NODOI":
            lock_string = f"{bk}|{entry['url']}|{entry['f3']}|{entry['f4']}"
        else:
            lock_string = f"{bk}|{doi_or_nodoi}|{entry['url']}|{entry['f3']}|{entry['f4']}"
        expected = sha256_hex(lock_string)
        if expected != sha:
            print(f"ERROR: sources.lock {bk} SHA mismatch\n  expected: {expected}\n  actual:   {sha}", file=sys.stderr)
            return 7

    print("OK: bibliography SSOT validated.")
    print(f"  Bibkeys: {len(bib)}")
    print(f"  Mapping entries: {len(mapping)}")
    print(f"  Claim IDs: {len(claim_ids)}")
    print(f"  Lock entries: {len(lock_rows)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

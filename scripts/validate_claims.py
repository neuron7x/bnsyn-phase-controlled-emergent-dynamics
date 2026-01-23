#!/usr/bin/env python3
import sys
import re
from pathlib import Path

REQUIRED_TOP_KEYS = {"version", "claims"}
REQUIRED_CLAIM_KEYS = {"id","statement","status","tier","normative","source","locator","action","notes"}

ALLOWED_STATUS = {"PROVEN","UNPROVEN"}
ALLOWED_TIER = {"A","B","C"}
ALLOWED_ACTION = {"KEEP","REMOVE","DOWNGRADE","CORRECT"}

def parse_yaml_minimal(text: str):
    """Very small YAML subset parser sufficient for this ledger format.
    Supports:
      - key: value
      - nested dicts
      - list under 'claims:' with '- key: value' items
    For full YAML, install PyYAML; this script avoids extra deps for CI.
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    data = {}
    i = 0
    def parse_value(v):
        v=v.strip()
        if v.lower() in ("true","false"):
            return v.lower()=="true"
        if v.startswith('"') and v.endswith('"'):
            return v[1:-1]
        return v
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("claims:"):
            i += 1
            claims=[]
            current=None
            while i < len(lines):
                ln = lines[i]
                if re.match(r"^[A-Za-z_].*:\s*$", ln):  # new top-level key
                    break
                if ln.lstrip().startswith("- "):
                    if current:
                        claims.append(current)
                    current={}
                    kv=ln.lstrip()[2:]
                    if ":" in kv:
                        k,v=kv.split(":",1)
                        current[k.strip()]=parse_value(v)
                else:
                    if ":" in ln:
                        k,v=ln.strip().split(":",1)
                        if current is None:
                            raise ValueError("Found claim field before list item")
                        current[k.strip()]=parse_value(v)
                i += 1
            if current:
                claims.append(current)
            data["claims"]=claims
            continue
        if ":" in ln:
            k,v=ln.split(":",1)
            data[k.strip()]=parse_value(v)
        i += 1
    return data

def fail(msg):
    print(f"[claims-gate] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)

def main():
    root = Path(__file__).resolve().parents[1]
    ledger = root / "claims" / "claims.yml"
    if not ledger.exists():
        fail("claims/claims.yml not found")

    raw = ledger.read_text(encoding="utf-8")
    try:
        y = parse_yaml_minimal(raw)
    except Exception as e:
        fail(f"Failed to parse claims.yml: {e}")

    missing = REQUIRED_TOP_KEYS - set(y.keys())
    if missing:
        fail(f"claims.yml missing top keys: {sorted(missing)}")

    claims = y.get("claims", [])
    if not isinstance(claims, list) or not claims:
        fail("claims.yml has empty or invalid 'claims' list")

    ids=set()
    for c in claims:
        if not isinstance(c, dict):
            fail("Claim entry is not a dict")
        miss = REQUIRED_CLAIM_KEYS - set(c.keys())
        if miss:
            fail(f"Claim {c.get('id','<no-id>')} missing keys: {sorted(miss)}")
        cid=str(c["id"]).strip()
        if not re.match(r"^CLM-\d{4}$", cid):
            fail(f"Invalid claim id format: {cid}")
        if cid in ids:
            fail(f"Duplicate claim id: {cid}")
        ids.add(cid)

        status=str(c["status"]).strip().upper()
        tier=str(c["tier"]).strip().upper()
        action=str(c["action"]).strip().upper()
        normative=bool(c["normative"])

        if status not in ALLOWED_STATUS:
            fail(f"{cid}: invalid status {status}")
        if tier not in ALLOWED_TIER:
            fail(f"{cid}: invalid tier {tier}")
        if action not in ALLOWED_ACTION:
            fail(f"{cid}: invalid action {action}")

        # Hard rule: NORMATIVE claims must be PROVEN + Tier-A and have source+locator
        if normative:
            if status != "PROVEN":
                fail(f"{cid}: normative=true requires status=PROVEN")
            if tier != "A":
                fail(f"{cid}: normative=true requires tier=A")
            if not str(c["source"]).strip() or str(c["source"]).strip().lower()=="unresolved":
                fail(f"{cid}: normative=true requires source")
            if not str(c["locator"]).strip() or str(c["locator"]).strip().lower()=="unresolved":
                fail(f"{cid}: normative=true requires locator")

    print(f"[claims-gate] OK: {len(claims)} claims validated; {sum(1 for c in claims if c.get('normative'))} normative.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

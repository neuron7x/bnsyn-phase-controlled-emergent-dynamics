"""Scan governed docs for claim bindings."""

from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.doc_contracts import extract_data, load_contract  # noqa: E402
CLAIMS = ROOT / "claims" / "claims.yml"
INVENTORY = ROOT / "docs" / "INVENTORY.md"


def load_governed_docs() -> list[Path]:
    data = extract_data(load_contract(INVENTORY))
    docs = data.get("governed_docs")
    if not isinstance(docs, list) or not docs:
        raise SystemExit("INVENTORY.md governed_docs list is empty")
    return [ROOT / str(p) for p in docs]


def load_claims() -> dict[str, dict[str, str | bool]]:
    data = yaml.safe_load(CLAIMS.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    if "schema" in data and "data" in data and isinstance(data["data"], dict):
        data = data["data"]
    claims = data.get("claims", [])
    out: dict[str, dict[str, str | bool]] = {}
    for c in claims:
        cid = c.get("id")
        if isinstance(cid, str):
            out[cid] = {
                "tier": str(c.get("tier", "")),
                "normative": bool(c.get("normative", False)),
            }
    return out


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def main() -> int:
    claim_map = load_claims()
    governed_docs = load_governed_docs()
    malformed = []
    missing = []
    invalid_tier = []

    for f in governed_docs:
        if not f.exists():
            continue
        rf = rel(f)
        data = extract_data(load_contract(f))
        normative_ids = data.get("normative_claim_ids", [])
        if not normative_ids:
            continue
        if not isinstance(normative_ids, list):
            malformed.append((rf, 0, "normative_claim_ids must be a list"))
            continue
        for cid in normative_ids:
            claim = claim_map.get(str(cid))
            if not claim:
                missing.append((rf, 0, str(cid)))
                continue
            if claim["tier"] != "Tier-A" or not claim["normative"]:
                invalid_tier.append((rf, 0, str(cid), claim["tier"], claim["normative"]))

    if malformed:
        print("ERROR: malformed normative claim bindings:")
        for rf, ln, line in malformed[:50]:
            print(f"  {rf}:{ln}: {line}")
        return 2

    if missing:
        print("ERROR: normative claim bindings missing Claim IDs:")
        for rf, ln, cid in missing[:50]:
            print(f"  {rf}:{ln}: {cid}")
        return 3

    if invalid_tier:
        print("ERROR: normative claim bindings must point to Tier-A normative claims:")
        for rf, ln, cid, tier, normative in invalid_tier[:50]:
            print(f"  {rf}:{ln}: {cid} (tier={tier}, normative={normative})")
        return 4

    print("OK: normative claim binding scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

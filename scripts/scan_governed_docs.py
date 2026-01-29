#!/usr/bin/env python3
"""Scan governed docs for document contracts."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.doc_contracts import extract_data, load_contract  # noqa: E402
INVENTORY = ROOT / "docs" / "INVENTORY.md"


def load_governed_docs() -> list[str]:
    if not INVENTORY.exists():
        raise SystemExit(f"Missing INVENTORY.md: {INVENTORY}")
    data = extract_data(load_contract(INVENTORY))
    docs = data.get("governed_docs")
    if not isinstance(docs, list) or not docs:
        raise SystemExit("INVENTORY.md governed_docs list is empty")
    return [str(p) for p in docs]


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def main() -> int:
    governed_docs = load_governed_docs()

    missing_files: list[str] = []
    scanned_files = 0

    for doc in governed_docs:
        path = ROOT / doc
        if not path.exists():
            missing_files.append(doc)
            continue
        scanned_files += 1

    print(f"[governed-docs] Governed docs listed: {len(governed_docs)}")
    print(f"[governed-docs] Files scanned: {scanned_files}")
    print(f"[governed-docs] Missing governed files: {len(missing_files)}")

    if missing_files:
        print("\nERROR: Missing governed files:")
        for doc in missing_files[:20]:
            print(f"  {doc}")
        return 1

    print("[governed-docs] OK: governed docs present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

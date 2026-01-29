from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sys

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.doc_contracts import SCHEMA_PATH, load_contract  # noqa: E402
TARGET_DIRS = [
    ROOT / "docs",
    ROOT / "specs",
    ROOT / "claims",
]
TARGET_EXTS = {".md", ".rst", ".yml", ".yaml"}
EXTRA_FILES = [
    ROOT / "README.md",
    ROOT / "README_CLAIMS_GATE.md",
    ROOT / "GOVERNANCE_VERIFICATION_REPORT.md",
    ROOT / "QUALITY_INDEX.md",
]


def iter_target_files() -> list[Path]:
    files: list[Path] = []
    for directory in TARGET_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in TARGET_EXTS:
                files.append(path)
    for extra in EXTRA_FILES:
        if extra.exists():
            files.append(extra)
    resolved = sorted({path.resolve() for path in files})
    claims_ledger = ROOT / "claims" / "claims.yml"
    return [path for path in resolved if path != claims_ledger]


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_document(path: Path, validator: Draft202012Validator) -> list[str]:
    contract = load_contract(path)
    errors = [error.message for error in validator.iter_errors(contract)]
    rel = path.relative_to(ROOT).as_posix()
    if contract.get("document") != rel:
        errors.append("document field does not match file path")
    return errors


def main() -> int:
    schema = load_schema()
    validator = Draft202012Validator(schema)
    failures: dict[str, list[str]] = {}
    for path in iter_target_files():
        errors = validate_document(path, validator)
        if errors:
            failures[path.relative_to(ROOT).as_posix()] = errors
    if failures:
        lines = ["Document contract validation failed:"]
        for doc, errors in sorted(failures.items()):
            lines.append(f"- {doc}")
            for error in errors:
                lines.append(f"  - {error}")
        raise SystemExit("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "document_contract.schema.json"


def load_contract(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Document contract must be a mapping: {path}")
    return data


def extract_data(contract: dict[str, Any]) -> dict[str, Any]:
    data = contract.get("data")
    if isinstance(data, dict):
        return data
    return {}

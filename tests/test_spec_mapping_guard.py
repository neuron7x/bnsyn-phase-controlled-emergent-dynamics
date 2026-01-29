"""Guard checks that spec references exist for invariant identifiers."""

from __future__ import annotations

from pathlib import Path

from scripts.doc_contracts import extract_data, load_contract


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_tla_spec_contains_invariant_ids() -> None:
    tla_text = _read("specs/tla/BNsyn.tla")
    for token in ("INV-1", "INV-2", "INV-3", "GainClamp", "TemperatureBounds", "GateBounds"):
        assert token in tla_text, f"Missing TLA+ identifier: {token}"


def test_vcg_spec_contains_invariant_ids() -> None:
    vcg_contract = load_contract(Path("docs/VCG.md"))
    data = extract_data(vcg_contract)
    tokens = data.get("invariant_ids", [])
    for token in ("I1", "I2", "I3", "I4", "A1", "A2", "A3", "A4"):
        assert token in tokens, f"Missing VCG identifier: {token}"

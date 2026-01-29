"""
Guard tests for invariant label mappings in specs/docs.

These tests ensure the invariant labels and canonical names remain present
in the specification sources. They rely on simple text containment checks
for stability (no line-number assumptions).
"""

from __future__ import annotations

from pathlib import Path


def _read_text(relative_path: str) -> str:
    return Path(relative_path).read_text(encoding="utf-8")


def test_tla_invariant_labels_present() -> None:
    """TLA+ spec retains invariant labels and canonical names."""
    tla_text = _read_text("specs/tla/BNsyn.tla")
    cfg_text = _read_text("specs/tla/BNsyn.cfg")
    readme_text = _read_text("specs/tla/README.md")

    for label in ("INV-1", "INV-2", "INV-3"):
        assert label in tla_text, f"Missing {label} in specs/tla/BNsyn.tla"

    for name in ("GainClamp", "TemperatureBounds", "GateBounds"):
        assert name in tla_text, f"Missing {name} in specs/tla/BNsyn.tla"
        assert name in cfg_text, f"Missing {name} in specs/tla/BNsyn.cfg"
        assert name in readme_text, f"Missing {name} in specs/tla/README.md"


def test_vcg_invariant_labels_present() -> None:
    """VCG spec retains I1–I4 and A1–A4 invariant labels."""
    vcg_text = _read_text("docs/VCG.md")

    for label in ("I1", "I2", "I3", "I4"):
        assert label in vcg_text, f"Missing {label} in docs/VCG.md"

    for label in ("A1", "A2", "A3", "A4"):
        assert label in vcg_text, f"Missing {label} in docs/VCG.md"

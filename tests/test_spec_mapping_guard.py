from pathlib import Path


def test_tla_spec_mapping_contains_invariants() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = repo_root / "specs" / "tla" / "README.md"
    tla_spec = repo_root / "specs" / "tla" / "BNsyn.tla"

    assert readme.exists()
    assert tla_spec.exists()

    readme_text = readme.read_text(encoding="utf-8")
    spec_text = tla_spec.read_text(encoding="utf-8")

    required_tokens = ["INV-1", "GainClamp", "INV-2", "TemperatureBounds", "INV-3", "GateBounds"]
    for token in required_tokens:
        assert token in readme_text
        assert token in spec_text

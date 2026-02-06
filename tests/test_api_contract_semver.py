from __future__ import annotations

from scripts.check_api_contract import check_api_changes, semver_allows_breaking_change


def test_api_contract_detects_breaking_changes() -> None:
    baseline = {"bnsyn.demo": {"f": "(a, b)"}}
    current = {"bnsyn.demo": {"f": "(a)"}}
    ok, breaking = check_api_changes(baseline, current)
    assert not ok
    assert any("Signature changed" in item for item in breaking)


def test_semver_major_bump_allows_breaking_change() -> None:
    assert semver_allows_breaking_change("0.2.0", "1.0.0")
    assert not semver_allows_breaking_change("0.2.0", "0.3.0")

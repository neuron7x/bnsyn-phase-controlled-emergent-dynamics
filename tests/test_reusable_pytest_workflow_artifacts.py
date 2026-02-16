from __future__ import annotations

from pathlib import Path


WORKFLOW_PATH = Path('.github/workflows/_reusable_pytest.yml')


def _workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding='utf-8')


def test_pytest_step_uses_pipefail_and_pipestatus() -> None:
    text = _workflow_text()
    assert 'set -o pipefail' in text
    assert 'PYTEST_EXIT=${PIPESTATUS[0]}' in text


def test_coverage_xml_upload_is_conditional() -> None:
    text = _workflow_text()
    assert "if: always() && hashFiles('coverage.xml') != ''" in text
    assert 'coverage.xml missing; skipping coverage-xml artifact upload' in text
    assert 'coverage.xml is required for smoke/unit coverage jobs' not in text

from __future__ import annotations

from pathlib import Path


def test_pr_description_policy_accepts_stable_aliases() -> None:
    workflow = Path('.github/workflows/ci-pr-atomic.yml').read_text(encoding='utf-8').lower()

    assert "const headings = new set(" in workflow
    assert "'risk': ['risk', 'risks', 'breaking changes', 'performance impact', 'additional notes', 'ризик', 'ризики']" in workflow
    assert "'evidence': ['evidence', 'testing', 'how to test', 'validation', 'tests', 'verification', 'доказ', 'докази']" in workflow
    assert "'how to test': ['how to test', 'testing', 'test plan', 'commands run', 'як тестувати', 'перевірка']" in workflow
    assert "const hasalias = (alias) => body.includes(alias)" in workflow

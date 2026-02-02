"""Tests for documentation link checker."""

from __future__ import annotations

from pathlib import Path

from tools.check_docs_links import check_docs_links


def test_docs_link_check_passes_for_valid_links(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()

    target = root / "target.md"
    target.write_text("# Target\n\n## Details\n", encoding="utf-8")

    source = root / "index.md"
    source.write_text("See [target](target.md#details).", encoding="utf-8")

    errors = check_docs_links(root, repo_root=tmp_path)
    assert errors == []


def test_docs_link_check_reports_missing_target(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()

    source = root / "index.md"
    source.write_text("Broken [link](missing.md).", encoding="utf-8")

    errors = check_docs_links(root, repo_root=tmp_path)
    assert errors

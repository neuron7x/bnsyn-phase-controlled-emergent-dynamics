from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = (REPO_ROOT / "src" / "bnsyn", REPO_ROOT / "scripts")


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for directory in TARGET_DIRS:
        files.extend(sorted(directory.glob("**/*.py")))
    return files


def test_no_bare_except_handlers_in_production_paths() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                violations.append(f"{path}:{node.lineno}")
    assert violations == []


def test_no_todo_or_fixme_in_production_paths() -> None:
    violations: list[str] = []
    needles = ("TODO", "FIXME")
    for path in _iter_python_files():
        content = path.read_text(encoding="utf-8")
        for index, line in enumerate(content.splitlines(), start=1):
            if any(needle in line for needle in needles):
                violations.append(f"{path}:{index}")
    assert violations == []

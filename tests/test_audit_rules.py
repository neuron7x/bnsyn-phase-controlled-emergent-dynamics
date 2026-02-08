from __future__ import annotations
import fnmatch
from pathlib import Path
import yaml


def _rules() -> dict:
    return yaml.safe_load(Path('audit/classification_rules.yml').read_text())


def _classify(path: str, rules: dict) -> str:
    for cat in ['GENERATED','CI_CD','INFRASTRUCTURE','TESTS','DATA_SCHEMAS','CORE_LOGIC','SCRIPTS_TOOLING','DOCUMENTATION','STATIC_ASSETS','CONFIGURATION']:
        for pattern in rules['categories'].get(cat, []):
            if fnmatch.fnmatch(path, pattern):
                return cat
    return 'CONFIGURATION'


def test_workflow_files_are_ci_cd() -> None:
    rules = _rules()
    wf = Path('.github/workflows')
    files = [p.as_posix() for p in wf.rglob('*') if p.is_file() and p.suffix in {'.yml', '.yaml'}]
    assert files, 'expected workflow files in repo'
    for f in files:
        assert _classify(f, rules) == 'CI_CD'


def test_excludes_do_not_hide_github_workflows() -> None:
    rules = _rules()
    patterns = [e['pattern'] for e in rules['excludes']]
    sample = '.github/workflows/ci-pr.yml'
    assert not any(fnmatch.fnmatch(sample, p) for p in patterns)

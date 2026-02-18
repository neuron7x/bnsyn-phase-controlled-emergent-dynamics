from __future__ import annotations

import importlib
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_policy_schema_and_deterministic_commands() -> None:
    policy = yaml.safe_load((REPO_ROOT / '.github' / 'ci_policy.yml').read_text(encoding='utf-8'))
    assert isinstance(policy, dict)
    assert policy['protocol'] == 'DE-TF-2026.03'

    p0 = set(policy['tiers']['P0'])
    for gate in ('ruff', 'mypy', 'pytest', 'build'):
        assert gate in p0
        cmd = policy['tools'][gate]['cmd']
        assert cmd.startswith('python -m ')


def test_p0_modules_importable_for_python_tools() -> None:
    policy = yaml.safe_load((REPO_ROOT / '.github' / 'ci_policy.yml').read_text(encoding='utf-8'))
    p0 = set(policy['tiers']['P0'])
    for name, tool in policy['tools'].items():
        if name not in p0:
            continue
        importlib.import_module(str(tool['module']).replace('-', '_'))

#!/usr/bin/env python3
from __future__ import annotations

import fnmatch
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'math_audit'


def load_rules() -> dict:
    return yaml.safe_load((ROOT / 'audit' / 'classification_rules.yml').read_text(encoding='utf-8'))


def git_files() -> list[str]:
    proc = subprocess.run(['git', 'ls-files'], cwd=ROOT, text=True, capture_output=True, check=True)
    return [line for line in proc.stdout.splitlines() if line]


def is_excluded(path: str, rules: dict) -> bool:
    return any(fnmatch.fnmatch(path, item['pattern']) for item in rules.get('excludes', []))


def classify(path: str, rules: dict) -> str:
    order = [
        'GENERATED',
        'CI_CD',
        'INFRASTRUCTURE',
        'TESTS',
        'DATA_SCHEMAS',
        'CORE_LOGIC',
        'SCRIPTS_TOOLING',
        'DOCUMENTATION',
        'STATIC_ASSETS',
        'CONFIGURATION',
    ]
    for category in order:
        for pattern in rules['categories'].get(category, []):
            if fnmatch.fnmatch(path, pattern):
                return category
    return 'CONFIGURATION'


def line_counts(path: Path) -> tuple[int, int, int]:
    try:
        lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    except Exception:
        return (0, 0, 0)
    code = comments = blank = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank += 1
        elif stripped.startswith(('#', '//', '/*', '*', '--')):
            comments += 1
        else:
            code += 1
    return (code, comments, blank)


def main() -> None:
    rules = load_rules()
    files = [p for p in git_files() if not is_excluded(p, rules)]

    categories = {
        key: {'files': [], 'files_count': 0, 'loc_code': 0, 'loc_comments': 0, 'loc_blank': 0}
        for key in ['CORE_LOGIC', 'TESTS', 'CI_CD', 'INFRASTRUCTURE', 'DOCUMENTATION', 'SCRIPTS_TOOLING', 'DATA_SCHEMAS', 'CONFIGURATION', 'STATIC_ASSETS', 'GENERATED']
    }
    for rel in sorted(files):
        category = classify(rel, rules)
        categories[category]['files'].append(rel)
        code, comments, blank = line_counts(ROOT / rel)
        categories[category]['loc_code'] += code
        categories[category]['loc_comments'] += comments
        categories[category]['loc_blank'] += blank

    for info in categories.values():
        info['files_count'] = len(info['files'])

    out = {'inventory_total_files': len(files), 'categories': categories}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'crosscheck_metrics.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()

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
    return yaml.safe_load((ROOT / 'audit' / 'classification_rules.yml').read_text())


def git_files() -> list[str]:
    p = subprocess.run(['git', 'ls-files'], cwd=ROOT, text=True, capture_output=True, check=True)
    return [x for x in p.stdout.splitlines() if x]


def excluded(path: str, rules: dict) -> bool:
    for e in rules.get('excludes', []):
        if fnmatch.fnmatch(path, e['pattern']):
            return True
    return False


def classify(path: str, rules: dict) -> str:
    order = ['GENERATED','CI_CD','INFRASTRUCTURE','TESTS','DATA_SCHEMAS','CORE_LOGIC','SCRIPTS_TOOLING','DOCUMENTATION','STATIC_ASSETS','CONFIGURATION']
    for cat in order:
        for pat in rules['categories'].get(cat, []):
            if fnmatch.fnmatch(path, pat):
                return cat
    return 'CONFIGURATION'


def count_code_only(path: Path) -> int:
    try:
        lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    except Exception:
        return 0
    c = 0
    for ln in lines:
        t = ln.strip()
        if not t or t.startswith(('#','//','/*','*','--')):
            continue
        c += 1
    return c


def main() -> None:
    rules = load_rules()
    files = [f for f in git_files() if not excluded(f, rules)]
    cats = {k: {'files_count': 0, 'loc_code': 0} for k in ['CORE_LOGIC','TESTS','CI_CD','INFRASTRUCTURE','DOCUMENTATION','SCRIPTS_TOOLING','DATA_SCHEMAS','CONFIGURATION','STATIC_ASSETS','GENERATED']}
    for rel in files:
        cat = classify(rel, rules)
        cats[cat]['files_count'] += 1
        cats[cat]['loc_code'] += count_code_only(ROOT / rel)
    out = {
        'inventory_total_files': len(files),
        'categories': cats,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'crosscheck_metrics.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()

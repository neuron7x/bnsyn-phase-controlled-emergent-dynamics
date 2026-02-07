#!/usr/bin/env python3
from __future__ import annotations
import ast, fnmatch, hashlib, json, os, re, subprocess
from collections import defaultdict
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'math_audit'
RAW = OUT / 'raw'


def run(cmd: list[str], name: str) -> str:
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f'{name}.stdout.log').write_text(p.stdout)
    (RAW / f'{name}.stderr.log').write_text(p.stderr)
    if p.returncode != 0:
        raise SystemExit(f'command failed: {cmd} rc={p.returncode}')
    return p.stdout


def load_rules() -> dict:
    return yaml.safe_load((ROOT / 'audit' / 'classification_rules.yml').read_text())


def is_excluded(path: str, rules: dict) -> bool:
    return any(fnmatch.fnmatch(path, p['pattern']) for p in rules.get('excludes', []))


def classify(path: str, rules: dict) -> str:
    for cat in ['GENERATED','CI_CD','INFRASTRUCTURE','TESTS','DATA_SCHEMAS','CORE_LOGIC','SCRIPTS_TOOLING','DOCUMENTATION','STATIC_ASSETS','CONFIGURATION']:
        for pattern in rules['categories'].get(cat, []):
            if fnmatch.fnmatch(path, pattern):
                return cat
    return 'CONFIGURATION'


def line_counts(path: Path) -> tuple[int,int,int]:
    code = comments = blank = 0
    try:
        text = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    except Exception:
        return (0,0,0)
    for line in text:
        t = line.strip()
        if not t:
            blank += 1
        elif t.startswith(('#','//','/*','*','--')):
            comments += 1
        else:
            code += 1
    return code, comments, blank


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    rules = load_rules()

    files = run(['git','ls-files'], 'git_ls_files').splitlines()
    files = [f for f in files if f and not is_excluded(f, rules)]

    categories: dict[str, dict] = {k: {'files': [], 'files_count': 0, 'loc_code': 0, 'loc_comments': 0, 'loc_blank': 0} for k in [
        'CORE_LOGIC','TESTS','CI_CD','INFRASTRUCTURE','DOCUMENTATION','SCRIPTS_TOOLING','DATA_SCHEMAS','CONFIGURATION','STATIC_ASSETS','GENERATED']}

    inventory = []
    ext_counts = defaultdict(int)
    for rel in sorted(files):
        p = ROOT / rel
        digest = sha256_file(p)
        inventory.append({'path': rel, 'sha256': digest, 'bytes': p.stat().st_size})
        ext_counts[p.suffix.lower() or '<none>'] += 1
        cat = classify(rel, rules)
        c, cm, b = line_counts(p)
        categories[cat]['files'].append(rel)
        categories[cat]['loc_code'] += c
        categories[cat]['loc_comments'] += cm
        categories[cat]['loc_blank'] += b

    for v in categories.values():
        v['files_count'] = len(v['files'])

    test_files = categories['TESTS']['files']
    test_funcs = negative_asserts = property_tests = 0
    for rel in test_files:
        p = ROOT / rel
        txt = p.read_text(encoding='utf-8', errors='ignore')
        negative_asserts += len(re.findall(r'pytest\.raises|assertRaises', txt))
        property_tests += len(re.findall(r'@given|hypothesis', txt))
        if p.suffix == '.py':
            try:
                t = ast.parse(txt)
                test_funcs += sum(1 for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name.startswith('test_'))
            except SyntaxError:
                pass

    shortlog = run(['git','shortlog','-sn','--all'], 'git_shortlog')
    contributors = []
    for line in shortlog.splitlines():
        m = re.match(r'\s*(\d+)\s+(.+)$', line)
        if m:
            contributors.append({'commits': int(m.group(1)), 'name': m.group(2)})

    metadata = {
        'head': run(['git','rev-parse','HEAD'], 'git_head').strip(),
        'branch': run(['git','branch','--show-current'], 'git_branch').strip(),
        'latest_commit': run(['git','log','-1','--date=iso-strict'], 'git_log_1').strip(),
        'first_commit': run(['git','log','--reverse','--format=%H %ai %an','-1'], 'git_log_first').strip(),
        'total_commits': int(run(['git','rev-list','--count','HEAD'], 'git_rev_list_count').strip()),
        'active_days': int(run(['bash','-lc',"git log --format='%ai' | awk '{print $1}' | sort -u | wc -l"], 'git_active_days').strip()),
        'contributors': contributors,
    }

    radon_raw = run(['.venv_audit/bin/radon','cc','.','-a','-s','--exclude=test_*,venv,node_modules,__pycache__,.git'], 'radon_cc')
    avg_match = re.search(r'Average complexity:\s*([A-F]) \(([^)]+)\)', radon_raw)
    grade_dist = defaultdict(int)
    for g in re.findall(r' - ([A-F]) \(', radon_raw):
        grade_dist[g] += 1

    metrics = {
        'metadata': metadata,
        'inventory': {'total_files': len(files), 'files': inventory},
        'categories': categories,
        'totals': {
            'countable_files': sum(v['files_count'] for k, v in categories.items() if k != 'GENERATED'),
            'countable_loc_code': sum(v['loc_code'] for k, v in categories.items() if k != 'GENERATED'),
            'countable_loc_comments': sum(v['loc_comments'] for k, v in categories.items() if k != 'GENERATED'),
            'countable_loc_blank': sum(v['loc_blank'] for k, v in categories.items() if k != 'GENERATED'),
        },
        'tests': {
            'test_files': len(test_files),
            'test_functions_ast': test_funcs,
            'negative_asserts': negative_asserts,
            'property_markers': property_tests,
        },
        'complexity': {
            'radon_average_grade': avg_match.group(1) if avg_match else 'UNKNOWN',
            'radon_average_score': float(avg_match.group(2)) if avg_match else None,
            'radon_grade_distribution': dict(sorted(grade_dist.items())),
        },
        'language_extensions': dict(sorted(ext_counts.items())),
    }

    (OUT / 'computed_metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()

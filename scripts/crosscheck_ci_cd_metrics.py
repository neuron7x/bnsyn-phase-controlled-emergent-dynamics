#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'math_audit'


def git_files() -> list[str]:
    p = subprocess.run(
        ['git', 'ls-files', '.github/workflows/*.yml', '.github/workflows/*.yaml'],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [x for x in p.stdout.splitlines() if x]


def main() -> None:
    workflows = sorted(git_files())
    jobs = 0
    steps = 0
    uses = set()
    for wf in workflows:
        txt = (ROOT / wf).read_text(encoding='utf-8', errors='ignore')
        jobs += len(re.findall(r'^\s{2}[A-Za-z0-9_-]+:\s*$', txt, flags=re.M))
        steps += len(re.findall(r'^\s*-\s+(name:|uses:|run:)', txt, flags=re.M))
        for m in re.finditer(r'\buses:\s*([^\n#]+)', txt):
            uses.add(m.group(1).strip())
    out = {
        'workflow_count': len(workflows),
        'job_count_proxy': jobs,
        'step_count_proxy': steps,
        'uses_actions_unique': sorted(uses),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'crosscheck_ci_cd_metrics.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()

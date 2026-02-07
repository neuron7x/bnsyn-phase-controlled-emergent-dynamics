#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import yaml

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
    OUT.mkdir(parents=True, exist_ok=True)
    workflows = sorted(git_files())
    actions = Counter()
    job_count = 0
    step_count = 0
    for wf in workflows:
        data = yaml.safe_load((ROOT / wf).read_text(encoding='utf-8', errors='ignore')) or {}
        jobs = data.get('jobs') or {}
        if isinstance(jobs, dict):
            job_count += len(jobs)
            for _, job in jobs.items():
                steps = (job or {}).get('steps') if isinstance(job, dict) else None
                if isinstance(steps, list):
                    step_count += len(steps)
                    for step in steps:
                        if isinstance(step, dict) and step.get('uses'):
                            actions[str(step['uses'])] += 1

    out = {
        'workflow_count': len(workflows),
        'workflow_files': workflows,
        'job_count': job_count,
        'step_count': step_count,
        'uses_actions_unique': sorted(actions.keys()),
        'uses_actions_frequency': actions.most_common(),
    }
    (OUT / 'ci_cd_metrics.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from jsonschema import validate


def test_work_volume_time_report_deterministic_and_valid() -> None:
    subprocess.run(['python', 'scripts/generate_work_volume_time_report.py'], check=True)
    first = Path('artifacts/work_volume_time/report.json').read_text(encoding='utf-8')
    subprocess.run(['python', 'scripts/generate_work_volume_time_report.py'], check=True)
    second = Path('artifacts/work_volume_time/report.json').read_text(encoding='utf-8')
    assert first == second

    schema = json.loads(Path('schemas/work_volume_time_report.schema.json').read_text(encoding='utf-8'))
    report = json.loads(second)
    validate(report, schema)

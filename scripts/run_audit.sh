#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

python scripts/generate_work_volume_time_report.py
python scripts/scan_unicode_controls.py
python - <<'PY'
import json
from pathlib import Path
from jsonschema import validate

schema = json.loads(Path('schemas/work_volume_time_report.schema.json').read_text(encoding='utf-8'))
report = json.loads(Path('artifacts/work_volume_time/report.json').read_text(encoding='utf-8'))
validate(report, schema)
print('WORK_VOLUME_TIME_REPORT_OK')
PY

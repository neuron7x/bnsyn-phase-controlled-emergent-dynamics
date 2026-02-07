#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'math_audit' / 'sources'


def slug(url: str) -> str:
    parsed = urlparse(url)
    base = (parsed.netloc + parsed.path).replace('/', '_').replace('.', '_')
    return re.sub(r'[^A-Za-z0-9_]+', '_', base).strip('_')[:120]


def extract_value(spec: dict, body: str) -> str | None:
    extraction = spec.get('extract', {})
    if extraction.get('type') == 'regex':
        match = re.search(extraction.get('pattern', ''), body)
        return match.group(1) if match and match.groups() else (match.group(0) if match else None)
    if extraction.get('type') == 'json':
        try:
            current = json.loads(body)
            for part in extraction.get('path', '').split('.'):
                current = current[part]
            return str(current)
        except Exception:
            return None
    return None


def main() -> None:
    cfg = yaml.safe_load((ROOT / 'audit_inputs' / 'sources.yml').read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    extracted = []

    for item in cfg.get('sources', []):
        url = item['url']
        tag = slug(url)
        record: dict[str, object] = {
            'id': item['id'],
            'url': url,
            'slug': tag,
            'unit': item.get('extract', {}).get('unit', 'UNKNOWN'),
            'status_code': None,
            'extracted_value': None,
            'error': None,
        }
        try:
            response = requests.get(url, timeout=30)
            body = response.text
            record['status_code'] = response.status_code
            (OUT / f'{tag}.html').write_text(body, encoding='utf-8')
            (OUT / f'{tag}.headers.txt').write_text('\n'.join(f'{k}: {v}' for k, v in response.headers.items()), encoding='utf-8')
            digest = hashlib.sha256(body.encode('utf-8', errors='ignore')).hexdigest()
            (OUT / f'{tag}.sha256').write_text(f'{digest}  {tag}.html\n', encoding='utf-8')
            record['extracted_value'] = extract_value(item, body)
        except Exception as exc:
            record['error'] = str(exc)
        extracted.append(record)

    payload = {'sources': sorted(extracted, key=lambda x: x['id'])}
    (ROOT / 'artifacts' / 'math_audit' / 'sources_extracted.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

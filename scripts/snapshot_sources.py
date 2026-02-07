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
    p = urlparse(url)
    base = (p.netloc + p.path).replace('/', '_').replace('.', '_')
    return re.sub(r'[^A-Za-z0-9_]+', '_', base).strip('_')[:120]


def extract_value(spec: dict, body: str) -> str | None:
    ext = spec.get('extract', {})
    if ext.get('type') == 'regex':
        m = re.search(ext.get('pattern', ''), body)
        return m.group(1) if m and m.groups() else (m.group(0) if m else None)
    if ext.get('type') == 'json':
        try:
            obj = json.loads(body)
            cur = obj
            for part in ext.get('path', '').split('.'):
                cur = cur[part]
            return str(cur)
        except Exception:
            return None
    return None


def main() -> None:
    cfg = yaml.safe_load((ROOT / 'audit_inputs' / 'sources.yml').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    extracted = []
    for item in cfg.get('sources', []):
        u = item['url']
        s = slug(u)
        r = requests.get(u, timeout=30)
        body = r.text
        (OUT / f'{s}.html').write_text(body)
        (OUT / f'{s}.headers.txt').write_text('\n'.join(f'{k}: {v}' for k, v in r.headers.items()))
        digest = hashlib.sha256(body.encode('utf-8', errors='ignore')).hexdigest()
        (OUT / f'{s}.sha256').write_text(f'{digest}  {s}.html\n')
        val = extract_value(item, body)
        extracted.append({
            'id': item['id'], 'url': u, 'slug': s, 'status_code': r.status_code,
            'extracted_value': val, 'unit': item.get('extract', {}).get('unit', 'UNKNOWN')
        })
    (ROOT / 'artifacts' / 'math_audit' / 'sources_extracted.json').write_text(json.dumps({'sources': extracted}, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()

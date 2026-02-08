#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from lxml import etree


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_claims(odt: Path, source_name: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(odt) as zf:
        content = zf.read('content.xml')
    root = etree.fromstring(content)
    ns = {'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}
    paras = ["".join(p.itertext()).strip() for p in root.xpath('.//text:p', namespaces=ns)]

    claims: list[dict[str, Any]] = []
    idx = 1
    for pidx, para in enumerate(paras, start=1):
        if not para:
            continue
        for m in re.finditer(r'(?<!\w)(\d[\d\s,\.]*)', para):
            val = m.group(1).strip()
            claims.append(
                {
                    'id': f'{source_name}-CLAIM-{idx:04d}',
                    'value': val,
                    'unit': 'UNKNOWN',
                    'context': para[:300],
                    'locator': f'content.xml:text:p[{pidx}]@{m.start()}-{m.end()}',
                }
            )
            idx += 1
    return claims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--output', default='audit_inputs/claims.json')
    ap.add_argument('--hash-dir', default='audit_inputs/odt_sources')
    args = ap.parse_args()

    hash_dir = Path(args.hash_dir)
    hash_dir.mkdir(parents=True, exist_ok=True)

    sources = []
    all_claims: list[dict[str, Any]] = []

    for in_path in args.inputs:
        p = Path(in_path)
        if not p.exists():
            continue
        digest = sha256_file(p)
        src_name = p.name
        (hash_dir / f'{src_name}.sha256').write_text(f'{digest}  {src_name}\n')
        sources.append({'name': src_name, 'sha256': digest})
        all_claims.extend(extract_claims(p, src_name))

    all_claims.sort(key=lambda x: x['id'])
    out = {'sources': sorted(sources, key=lambda x: x['name']), 'claims': all_claims}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

TEXT_NS = {'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract_claims_from_odt(path: Path, source_name: str) -> dict[str, dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        content = archive.read('content.xml')
    root = ET.fromstring(content)
    claims: dict[str, dict[str, str]] = {}
    index = 1
    for pidx, paragraph in enumerate(root.findall('.//text:p', TEXT_NS), start=1):
        text = ''.join(paragraph.itertext()).strip()
        if not text:
            continue
        for match in re.finditer(r'(?<!\w)(\d[\d\s,\.]*)', text):
            key = f'{source_name}-CLAIM-{index:04d}'
            claims[key] = {
                'value': match.group(1).strip(),
                'unit': 'UNKNOWN',
                'context': text[:300],
                'locator': f'content.xml:text:p[{pidx}]@{match.start()}-{match.end()}',
            }
            index += 1
    return claims


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='*', default=[])
    parser.add_argument('--output', default='audit_inputs/claims.json')
    parser.add_argument('--hash-dir', default='audit_inputs/odt_sources')
    args = parser.parse_args()

    hash_dir = Path(args.hash_dir)
    hash_dir.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, str]] = []
    claims: dict[str, dict[str, str]] = {}

    for candidate in sorted({str(Path(raw)) for raw in args.inputs}):
        path = Path(candidate)
        if not path.exists() or path.suffix.lower() != '.odt':
            continue
        source_name = path.name
        file_hash = sha256_file(path)
        (hash_dir / f'{source_name}.sha256').write_text(f'{file_hash}  {source_name}\n', encoding='utf-8')
        sources.append({'name': source_name, 'sha256': file_hash})
        claims.update(extract_claims_from_odt(path, source_name))

    if claims:
        payload = {'claims': dict(sorted(claims.items())), 'sources': sorted(sources, key=lambda x: x['name'])}
    else:
        payload = {'claims': {}, 'sources': sorted(sources, key=lambda x: x['name']), 'note': 'No external ODT inputs provided'}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

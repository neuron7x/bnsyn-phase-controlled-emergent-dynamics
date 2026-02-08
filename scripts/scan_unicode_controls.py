#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXT = {'.md', '.py', '.yml', '.yaml', '.json', '.txt'}
BIDI_POINTS = set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A))


def tracked_files() -> list[str]:
    p = subprocess.run(['git', 'ls-files'], cwd=ROOT, text=True, capture_output=True, check=True)
    return [line for line in p.stdout.splitlines() if line]


def main() -> None:
    bad: list[str] = []
    for rel in tracked_files():
        path = ROOT / rel
        if path.suffix.lower() not in TEXT_EXT:
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except Exception:
            continue
        for idx, ch in enumerate(text):
            if ord(ch) in BIDI_POINTS:
                bad.append(f'{rel}:{idx}:U+{ord(ch):04X}')
    if bad:
        print('Unicode control scan failed:')
        for item in bad:
            print(item)
        raise SystemExit(1)
    print('Unicode control scan passed')


if __name__ == '__main__':
    main()

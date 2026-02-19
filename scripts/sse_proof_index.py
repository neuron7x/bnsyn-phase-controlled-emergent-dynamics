from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "sse_sdo"
QUALITY = ART / "07_quality" / "quality.json"
OUT = ART / "07_quality" / "EVIDENCE_INDEX.md"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    quality = json.loads(QUALITY.read_text(encoding="utf-8"))
    lines = ["# EVIDENCE_INDEX", ""]
    for gate in quality.get("gates", []):
        lines.append(f"## {gate['id']} {gate['name']}")
        lines.append(f"- cmd: `{gate['cmd']}`")
        for artifact in gate.get("artifacts", []):
            rel = Path(artifact)
            if (ROOT / rel).exists():
                lines.append(f"- §REF:blob:{artifact}#{sha256_file(ROOT / rel)}")
        lines.append("")
    OUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

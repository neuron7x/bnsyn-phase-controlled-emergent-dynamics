from __future__ import annotations

from pathlib import Path

REQUIRED_SECTIONS = [
    "## Detection signals",
    "## Triage checklist",
    "## Reproduction commands",
    "## Rollback/mitigation procedure",
    "## Known failure modes",
]


def validate_runbook(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8")
    errors: list[str] = []
    if not content.startswith("# RUNBOOK:"):
        errors.append("missing runbook title")

    for section in REQUIRED_SECTIONS:
        if section not in content:
            errors.append(f"missing section {section}")

    if "```bash" not in content:
        errors.append("missing bash command block")

    if "| Failure mode | Signal | Mitigation |" not in content:
        errors.append("missing known failure modes table header")

    return errors


def main() -> int:
    runbook_dir = Path("docs/ops/runbooks")
    paths = sorted(runbook_dir.glob("RUNBOOK_*.md"))
    if len(paths) < 4:
        print("ERROR: expected at least 4 runbooks")
        return 1

    had_errors = False
    for path in paths:
        errors = validate_runbook(path)
        if errors:
            had_errors = True
            for err in errors:
                print(f"ERROR [{path}]: {err}")

    if had_errors:
        return 1

    print(f"Validated {len(paths)} runbooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

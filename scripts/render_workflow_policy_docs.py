from __future__ import annotations

from pathlib import Path

BEGIN = "<!-- BEGIN: AUTO-GENERATED WORKFLOW POLICIES -->"
END = "<!-- END: AUTO-GENERATED WORKFLOW POLICIES -->"


def render_block() -> str:
    lines = [
        BEGIN,
        "## Workflow Policy Rules (Auto-generated)",
        "",
        "| Rule ID | Statement | Enforcement |",
        "| --- | --- | --- |",
        "| R1 | Workflows with Gate Class `long-running` MUST NOT declare `push` or `pull_request` triggers. | `python scripts/validate_long_running_triggers.py` (exit 0 OK, exit 2 violations, exit 3 parse errors). |",
        END,
    ]
    return "\n".join(lines) + "\n"


def update_document(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    begin_index = text.find(BEGIN)
    end_index = text.find(END)
    if begin_index == -1 or end_index == -1 or end_index < begin_index:
        raise SystemExit("AUTO-GENERATED WORKFLOW POLICIES markers missing")

    end_index += len(END)
    new_block = render_block().rstrip("\n")
    before = text[:begin_index].rstrip("\n")
    after = text[end_index:].lstrip("\n")
    updated = f"{before}\n\n{new_block}\n\n{after}".rstrip("\n") + "\n"
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> int:
    path = Path(".github/WORKFLOW_CONTRACTS.md")
    changed = update_document(path)
    if changed:
        print("Updated workflow policy documentation block.")
    else:
        print("Workflow policy documentation block already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

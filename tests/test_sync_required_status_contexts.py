from __future__ import annotations

from pathlib import Path

from scripts.sync_required_status_contexts import build_payload, sync_required_status_contexts


def test_build_payload_matches_contract() -> None:
    payload = build_payload(Path('.github/PR_GATES.yml'), Path('.github/workflows'))

    assert payload["version"] == "1"
    contexts = payload["required_status_contexts"]
    assert isinstance(contexts, list)
    assert contexts
    assert "ci-pr-atomic / gate-profile" in contexts


def test_sync_required_status_contexts_check_and_update(tmp_path: Path) -> None:
    contexts_path = tmp_path / "REQUIRED_STATUS_CONTEXTS.yml"
    contexts_path.write_text("version: '1'\nrequired_status_contexts: []\n", encoding="utf-8")

    check_exit = sync_required_status_contexts(
        contexts_path=contexts_path,
        pr_gates_path=Path('.github/PR_GATES.yml'),
        workflows_dir=Path('.github/workflows'),
        check=True,
    )
    assert check_exit == 2

    update_exit = sync_required_status_contexts(
        contexts_path=contexts_path,
        pr_gates_path=Path('.github/PR_GATES.yml'),
        workflows_dir=Path('.github/workflows'),
        check=False,
    )
    assert update_exit == 0

    saved = contexts_path.read_text(encoding="utf-8")
    assert "required_status_contexts:" in saved
    assert "ci-pr-atomic / gate-profile" in saved

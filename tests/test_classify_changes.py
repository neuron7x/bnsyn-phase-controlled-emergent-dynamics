from scripts.classify_changes import classify_files


def test_docs_only_classification() -> None:
    flags = classify_files(["docs/ci.md", "README.md", ".github/PR_TEMPLATE.md"])
    assert flags.docs is True
    assert flags.docs_only is True
    assert flags.unknown_changed is False


def test_unknown_file_forces_fail_closed() -> None:
    flags = classify_files(["random.bin"])
    assert flags.docs_only is False
    assert flags.unknown_changed is True


def test_code_and_dependencies_classification() -> None:
    flags = classify_files(["src/main.py", "tests/test_main.py", "requirements-lock.txt"])
    assert flags.code_changed is True
    assert flags.tests_changed is True
    assert flags.deps_changed is True
    assert flags.dependency_manifest is True
    assert flags.docs_only is False


def test_workflow_and_actions_are_sensitive() -> None:
    flags = classify_files([".github/workflows/ci.yml", ".github/actions/x/action.yml"])
    assert flags.workflows_changed is True
    assert flags.docs_only is False

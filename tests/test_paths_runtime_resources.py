from __future__ import annotations

from bnsyn.paths import package_file, runtime_file


def test_package_file_resolves_packaged_runtime_assets() -> None:
    path = package_file("configs/canonical_profile.yaml")
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip()


def test_runtime_file_resolves_schema_asset() -> None:
    path = runtime_file("schemas/run-manifest.schema.json")
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip().startswith("{")

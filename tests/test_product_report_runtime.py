from __future__ import annotations

from pathlib import Path

import pytest

from bnsyn.proof.contracts import manifest_self_hash
from bnsyn.viz.product_report import _render_index_html, _validate_product_summary


def test_manifest_self_hash_does_not_mutate_input() -> None:
    manifest = {
        "schema_version": "1.0.0",
        "artifacts": {
            "run_manifest.json": "x" * 64,
            "summary_metrics.json": "1" * 64,
        },
    }
    original = {
        "schema_version": manifest["schema_version"],
        "artifacts": dict(manifest["artifacts"]),
    }

    digest = manifest_self_hash(manifest)

    assert isinstance(digest, str)
    assert len(digest) == 64
    assert manifest == original


def test_validate_product_summary_fails_closed_on_bad_type() -> None:
    payload = {
        "status": "PASS",
        "profile": "canonical",
        "seed": "123",
        "artifact_dir": "artifacts/canonical_run",
        "primary_visual": "emergence_plot.png",
        "proof_verdict": "PASS",
        "criticality_verdict": True,
        "avalanche_verdict": True,
        "generated_at": "1970-01-01T00:00:00Z",
        "package_version": "0.2.0",
        "bundle_contract_version": "canonical-export-proof",
    }

    with pytest.raises(ValueError, match="invalid type"):
        _validate_product_summary(payload)  # type: ignore[arg-type]


def test_render_index_html_is_pretty_printed() -> None:
    html = _render_index_html(
        manifest={"artifacts": {"summary_metrics.json": "0" * 64}},
        summary={"rate_mean_hz": 1.0},
        product_summary={
            "status": "PASS",
            "profile": "canonical",
            "seed": 123,
            "artifact_dir": Path("artifacts/canonical_run").as_posix(),
            "primary_visual": "emergence_plot.png",
            "proof_verdict": "PASS",
            "criticality_verdict": True,
            "avalanche_verdict": True,
            "generated_at": "1970-01-01T00:00:00Z",
            "package_version": "0.2.0",
            "bundle_contract_version": "canonical-export-proof",
        },
    )

    assert "\n  <head>" in html
    assert "\n  <body>" in html

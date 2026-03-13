from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from bnsyn.proof.contracts import CANONICAL_EXPORT_PROOF_CONTRACT

_DETERMINISTIC_TIMESTAMP = "1970-01-01T00:00:00Z"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _proof_verdict(artifact_dir: Path) -> str:
    proof = _read_json(artifact_dir / "proof_report.json")
    verdict = proof.get("verdict")
    if not isinstance(verdict, str) or verdict not in {"PASS", "FAIL"}:
        raise ValueError("proof_report.json must include verdict PASS|FAIL")
    return verdict


def write_product_report_bundle(
    *, artifact_dir: Path, profile: str, seed: int, package_version: str
) -> dict[str, Path]:
    summary = _read_json(artifact_dir / "summary_metrics.json")
    manifest = _read_json(artifact_dir / "run_manifest.json")
    criticality = _read_json(artifact_dir / "criticality_report.json")
    avalanche = _read_json(artifact_dir / "avalanche_report.json")
    proof_verdict = _proof_verdict(artifact_dir)

    criticality_verdict: str | bool = bool(
        isinstance(criticality.get("sigma_within_band_fraction"), (int, float))
        and float(criticality["sigma_within_band_fraction"]) > 0.0
    )
    avalanche_verdict: str | bool = bool(int(avalanche.get("avalanche_count", 0)) > 0)

    product_summary: dict[str, Any] = {
        "status": proof_verdict,
        "profile": profile,
        "seed": int(seed),
        "artifact_dir": artifact_dir.as_posix(),
        "primary_visual": "emergence_plot.png",
        "proof_verdict": proof_verdict,
        "criticality_verdict": criticality_verdict,
        "avalanche_verdict": avalanche_verdict,
        "generated_at": _DETERMINISTIC_TIMESTAMP,
        "package_version": package_version,
        "bundle_contract_version": str(manifest.get("bundle_contract", "unknown")),
    }

    product_summary_path = artifact_dir / "product_summary.json"
    product_summary_path.write_text(
        json.dumps(product_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    index_html = _render_index_html(
        manifest=manifest,
        summary=summary,
        product_summary=product_summary,
    )
    index_html_path = artifact_dir / "index.html"
    index_html_path.write_text(index_html, encoding="utf-8")

    return {"product_summary": product_summary_path, "index_html": index_html_path}


def _render_index_html(
    *, manifest: dict[str, Any], summary: dict[str, Any], product_summary: dict[str, Any]
) -> str:
    metric_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in sorted(summary.items())
    )
    artifact_links = "".join(
        f"<li><a href=\"{html.escape(name)}\">{html.escape(name)}</a></li>"
        for name in sorted(manifest.get("artifacts", {}).keys())
    )

    contract = html.escape(str(product_summary.get("bundle_contract_version", "unknown")))
    package_version = html.escape(str(product_summary.get("package_version", "unknown")))
    status = html.escape(str(product_summary.get("status", "FAIL")))

    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head><meta charset=\"utf-8\"/><title>BN-Syn Canonical Product Report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;}"
        ".ok{color:#0a7f2e;font-weight:bold;}table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:left;}th{background:#f5f5f5;}</style></head>\n"
        "<body>"
        "<h1>BN-Syn Canonical Product Report</h1>"
        f"<p><strong>Run status:</strong> <span class=\"ok\">{status}</span></p>"
        "<p>This report proves a deterministic canonical run executed and emitted the expected evidence bundle; it does not imply unverified biological or cognitive claims.</p>"
        "<h2>Primary visualization</h2>"
        "<p><img src=\"emergence_plot.png\" alt=\"Emergence plot\" style=\"max-width:100%;border:1px solid #ddd;\"/></p>"
        "<h2>Verdict summary</h2>"
        f"<ul><li>Proof verdict: {html.escape(str(product_summary.get('proof_verdict')))}</li>"
        f"<li>Criticality verdict: {html.escape(str(product_summary.get('criticality_verdict')))}</li>"
        f"<li>Avalanche verdict: {html.escape(str(product_summary.get('avalanche_verdict')))}</li></ul>"
        "<h2>Key metrics</h2>"
        f"<table>{metric_rows}</table>"
        "<h2>Machine artifacts</h2>"
        f"<ul>{artifact_links}<li><a href=\"product_summary.json\">product_summary.json</a></li></ul>"
        "<h2>Provenance</h2>"
        f"<p>bundle_contract={contract}, package_version={package_version}, generated_at={_DETERMINISTIC_TIMESTAMP}</p>"
        "</body></html>\n"
    )


def canonical_bundle_contract_supported(contract: str) -> bool:
    return contract == CANONICAL_EXPORT_PROOF_CONTRACT

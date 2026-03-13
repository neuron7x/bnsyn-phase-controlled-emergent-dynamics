from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, get_type_hints
from xml.etree import ElementTree as ET

from bnsyn.proof.contracts import CANONICAL_EXPORT_PROOF_CONTRACT

_DETERMINISTIC_TIMESTAMP = "1970-01-01T00:00:00Z"


class ProductSummary(TypedDict):
    status: str
    profile: str
    seed: int
    artifact_dir: str
    primary_visual: str
    proof_verdict: str
    criticality_verdict: bool
    avalanche_verdict: bool
    generated_at: str
    package_version: str
    bundle_contract_version: str




def _validate_product_summary(summary: ProductSummary) -> None:
    expected_types = get_type_hints(ProductSummary)
    for key, expected in expected_types.items():
        value = summary.get(key)
        if value is None:
            raise ValueError(f"product_summary missing required field: {key}")
        if not isinstance(value, expected):
            raise ValueError(
                f"product_summary field {key} has invalid type: expected {expected}, got {type(value)}"
            )

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

    criticality_verdict = bool(
        isinstance(criticality.get("sigma_within_band_fraction"), (int, float))
        and float(criticality["sigma_within_band_fraction"]) > 0.0
    )
    avalanche_verdict = bool(int(avalanche.get("avalanche_count", 0)) > 0)

    product_summary: ProductSummary = {
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

    _validate_product_summary(product_summary)

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
    *, manifest: dict[str, Any], summary: dict[str, Any], product_summary: ProductSummary
) -> str:
    root = ET.Element("html", {"lang": "en"})
    head = ET.SubElement(root, "head")
    ET.SubElement(head, "meta", {"charset": "utf-8"})
    ET.SubElement(head, "title").text = "BN-Syn Canonical Product Report"
    ET.SubElement(head, "style").text = (
        "body{font-family:Arial,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;}"
        ".ok{color:#0a7f2e;font-weight:bold;}table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:left;}th{background:#f5f5f5;}"
    )

    body = ET.SubElement(root, "body")
    ET.SubElement(body, "h1").text = "BN-Syn Canonical Product Report"

    p_status = ET.SubElement(body, "p")
    ET.SubElement(p_status, "strong").text = "Run status:"
    p_status.text = (p_status.text or "") + " "
    ET.SubElement(p_status, "span", {"class": "ok"}).text = product_summary["status"]

    ET.SubElement(body, "p").text = (
        "This report proves a deterministic canonical run executed and emitted the expected "
        "evidence bundle; it does not imply unverified biological or cognitive claims."
    )

    ET.SubElement(body, "h2").text = "Primary visualization"
    p_image = ET.SubElement(body, "p")
    ET.SubElement(
        p_image,
        "img",
        {
            "src": "emergence_plot.png",
            "alt": "Emergence plot",
            "style": "max-width:100%;border:1px solid #ddd;",
        },
    )

    ET.SubElement(body, "h2").text = "Verdict summary"
    ul_verdict = ET.SubElement(body, "ul")
    ET.SubElement(ul_verdict, "li").text = f"Proof verdict: {product_summary['proof_verdict']}"
    ET.SubElement(ul_verdict, "li").text = f"Criticality verdict: {product_summary['criticality_verdict']}"
    ET.SubElement(ul_verdict, "li").text = f"Avalanche verdict: {product_summary['avalanche_verdict']}"

    ET.SubElement(body, "h2").text = "Key metrics"
    table = ET.SubElement(body, "table")
    for key, value in sorted(summary.items()):
        tr = ET.SubElement(table, "tr")
        ET.SubElement(tr, "th").text = str(key)
        ET.SubElement(tr, "td").text = str(value)

    ET.SubElement(body, "h2").text = "Machine artifacts"
    ul_artifacts = ET.SubElement(body, "ul")
    artifacts = manifest.get("artifacts", {})
    if isinstance(artifacts, dict):
        for name in sorted(artifacts.keys()):
            li = ET.SubElement(ul_artifacts, "li")
            ET.SubElement(li, "a", {"href": str(name)}).text = str(name)
    li_summary = ET.SubElement(ul_artifacts, "li")
    ET.SubElement(li_summary, "a", {"href": "product_summary.json"}).text = "product_summary.json"

    ET.SubElement(body, "h2").text = "Provenance"
    ET.SubElement(body, "p").text = (
        f"bundle_contract={product_summary['bundle_contract_version']}, "
        f"package_version={product_summary['package_version']}, "
        f"generated_at={_DETERMINISTIC_TIMESTAMP}"
    )

    ET.indent(root)
    html_string = ET.tostring(root, encoding="unicode", method="html")
    return "<!DOCTYPE html>\n" + html_string + "\n"


def canonical_bundle_contract_supported(contract: str) -> bool:
    return contract == CANONICAL_EXPORT_PROOF_CONTRACT

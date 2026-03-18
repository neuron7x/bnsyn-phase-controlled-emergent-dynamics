from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypedDict, get_type_hints
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
    primary_visual_sha256: str
    primary_visual_data_artifact: str
    primary_visual_data_sha256: str


class ArtifactGuideEntry(TypedDict):
    name: str
    label: str
    purpose: str
    group: Literal["primary", "secondary", "auxiliary"]


ARTIFACT_GUIDE: tuple[ArtifactGuideEntry, ...] = (
    {
        "name": "index.html",
        "label": "Primary report",
        "purpose": "Open this first for the human-readable verdict and navigation sequence.",
        "group": "primary",
    },
    {
        "name": "product_summary.json",
        "label": "Machine summary",
        "purpose": "Fast machine-readable verdict payload for downstream checks.",
        "group": "primary",
    },
    {
        "name": "proof_report.json",
        "label": "Proof verdict",
        "purpose": "Gate-level PASS/FAIL verdict for the canonical proof bundle.",
        "group": "primary",
    },
    {
        "name": "summary_metrics.json",
        "label": "Headline metrics",
        "purpose": "Top-level measurements that summarize the run.",
        "group": "secondary",
    },
    {
        "name": "criticality_report.json",
        "label": "Criticality evidence",
        "purpose": "Shows whether sigma and branching behavior remain admissible.",
        "group": "secondary",
    },
    {
        "name": "avalanche_report.json",
        "label": "Avalanche evidence",
        "purpose": "Summarizes avalanche counts and event-structure behavior.",
        "group": "secondary",
    },
    {
        "name": "phase_space_report.json",
        "label": "Phase-space evidence",
        "purpose": "Summarizes trajectory and occupancy behavior in state space.",
        "group": "secondary",
    },
    {
        "name": "emergence_plot.png",
        "label": "Primary visual",
        "purpose": "Fastest visual confirmation of emergent activity in the run.",
        "group": "secondary",
    },
    {
        "name": "phase_space_rate_sigma.png",
        "label": "Rate-sigma plot",
        "purpose": "Visual phase-space projection for rate versus sigma.",
        "group": "auxiliary",
    },
    {
        "name": "phase_space_rate_coherence.png",
        "label": "Rate-coherence plot",
        "purpose": "Visual phase-space projection for rate versus coherence.",
        "group": "auxiliary",
    },
    {
        "name": "phase_space_activity_map.png",
        "label": "Activity map",
        "purpose": "Shows where the system spends time in rate-sigma space.",
        "group": "auxiliary",
    },
    {
        "name": "population_rate_trace.npy",
        "label": "Rate trace",
        "purpose": "Raw population-rate evidence for downstream analysis.",
        "group": "auxiliary",
    },
    {
        "name": "sigma_trace.npy",
        "label": "Sigma trace",
        "purpose": "Raw sigma evidence for criticality analysis.",
        "group": "auxiliary",
    },
    {
        "name": "coherence_trace.npy",
        "label": "Coherence trace",
        "purpose": "Raw coherence evidence for synchrony analysis.",
        "group": "auxiliary",
    },
    {
        "name": "run_manifest.json",
        "label": "Reproducibility manifest",
        "purpose": "Hashes, command metadata, and bundle contract lineage.",
        "group": "auxiliary",
    },
    {
        "name": "robustness_report.json",
        "label": "Robustness sweep",
        "purpose": "Fixed 10-seed admissibility summary for repeat-run checks.",
        "group": "auxiliary",
    },
    {
        "name": "envelope_report.json",
        "label": "Envelope verdict",
        "purpose": "Admissibility-band summary for the fixed run set.",
        "group": "auxiliary",
    },
    {
        "name": "avalanche_fit_report.json",
        "label": "Avalanche fit quality",
        "purpose": "Validity evidence for the fitted avalanche model.",
        "group": "auxiliary",
    },
)

CANONICAL_EXECUTION_CYCLE: tuple[tuple[str, str], ...] = (
    (
        "Bootstrap and run",
        "Execute `make quickstart-smoke` for the deterministic first-run path, or run the raw canonical CLI when the environment is already prepared.",
    ),
    (
        "Inspect the human surface",
        "Open `index.html` first, then review `product_summary.json` and `proof_report.json` before drilling into metrics or traces.",
    ),
    (
        "Validate the bundle",
        "Re-run `bnsyn validate-bundle <artifact_dir>` so the product surface, proof gate, and manifest stay aligned.",
    ),
    (
        "Replay or compare locally",
        "Re-execute `bnsyn run --profile canonical --plot --export-proof --output <artifact_dir>` when you need a fresh local bundle for side-by-side review.",
    ),
)

MERGE_READINESS_SIGNALS: tuple[str, ...] = (
    "proof_report.json shows PASS",
    "index.html, product_summary.json, and run_manifest.json are present together",
    "manifest hashes anchor the primary visual and raw trace evidence",
    "`bnsyn validate-bundle <artifact_dir>` passes locally before merge",
)

FAQ_ENTRIES: tuple[tuple[str, str], ...] = (
    (
        "What should I open first?",
        "Open index.html first. It is the permanent primary interface for the canonical bundle.",
    ),
    (
        "What file gives the fastest machine verdict?",
        "Read product_summary.json for the compact machine summary, then proof_report.json for the gate verdict.",
    ),
    (
        "How do I validate the bundle again?",
        "Run `bnsyn validate-bundle <artifact_dir>` against this directory to re-check the canonical product bundle.",
    ),
    (
        "Where are the raw traces?",
        "Use population_rate_trace.npy, sigma_trace.npy, and coherence_trace.npy for raw deterministic evidence.",
    ),
)


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


def _manifest_artifact_sha(manifest: dict[str, Any], artifact_name: str) -> str:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("run_manifest.json must contain an artifacts mapping")
    digest = artifacts.get(artifact_name)
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"run_manifest.json missing valid sha256 for {artifact_name}")
    return digest


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
        "primary_visual_sha256": _manifest_artifact_sha(manifest, "emergence_plot.png"),
        "primary_visual_data_artifact": "population_rate_trace.npy",
        "primary_visual_data_sha256": _manifest_artifact_sha(manifest, "population_rate_trace.npy"),
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


def _artifact_entries_present(manifest: dict[str, Any]) -> list[ArtifactGuideEntry]:
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return []
    available = set(artifacts.keys()) | {"product_summary.json", "index.html"}
    return [entry for entry in ARTIFACT_GUIDE if entry["name"] in available]


def _render_index_html(
    *, manifest: dict[str, Any], summary: dict[str, Any], product_summary: ProductSummary
) -> str:
    root = ET.Element("html", {"lang": "en"})
    head = ET.SubElement(root, "head")
    ET.SubElement(head, "meta", {"charset": "utf-8"})
    ET.SubElement(head, "title").text = "BN-Syn Canonical Product Report"
    ET.SubElement(head, "style").text = (
        "body{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.5;}"
        ".ok{color:#0a7f2e;font-weight:bold;}"
        ".lead{font-size:1.05rem;background:#f7f9fc;padding:0.9rem 1rem;border-left:4px solid #2f6fed;}"
        ".callout{background:#fbfcfe;border:1px solid #dbe4f0;border-radius:8px;padding:1rem;margin:1rem 0;}"
        "table{border-collapse:collapse;width:100%;margin:0.75rem 0 1.5rem 0;}"
        "th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top;}"
        "th{background:#f5f5f5;}"
        "code{background:#f4f4f4;padding:0.1rem 0.3rem;border-radius:4px;}"
    )

    body = ET.SubElement(root, "body")
    ET.SubElement(body, "h1").text = "BN-Syn Canonical Product Report"

    lead = ET.SubElement(body, "p", {"class": "lead"})
    lead.text = (
        "This is the primary human interface for a canonical BN-Syn bundle. "
        "Start here, then follow the fixed inspection order below."
    )

    p_status = ET.SubElement(body, "p")
    ET.SubElement(p_status, "strong").text = "Run status:"
    p_status.text = (p_status.text or "") + " "
    ET.SubElement(p_status, "span", {"class": "ok"}).text = product_summary["status"]

    ET.SubElement(body, "p").text = (
        "This report proves a deterministic canonical run executed and emitted the expected "
        "evidence bundle; it does not imply unverified biological or cognitive claims."
    )

    callout = ET.SubElement(body, "div", {"class": "callout"})
    ET.SubElement(callout, "h2").text = "Open this report first"
    ET.SubElement(callout, "p").text = (
        "Primary interface fixation: keep human review anchored here before opening raw JSON or NPY artifacts."
    )
    ordered = ET.SubElement(callout, "ol")
    for entry in _artifact_entries_present(manifest):
        li = ET.SubElement(ordered, "li")
        anchor = ET.SubElement(li, "a", {"href": entry["name"]})
        anchor.text = entry["name"]
        anchor.tail = f" — {entry['label']}: {entry['purpose']}"

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
    ET.SubElement(body, "p").text = (
        "Cryptographic evidence link: emergence_plot.png is anchored to population_rate_trace.npy "
        "through the canonical run_manifest.json SHA-256 entries shown below."
    )

    ET.SubElement(body, "h2").text = "Canonical execution cycle"
    cycle_intro = ET.SubElement(body, "p")
    cycle_intro.text = (
        "Treat the canonical bundle as a closed local review loop: run it, inspect it, validate it, and rerun it "
        "without losing the fixed human-first inspection order."
    )
    cycle_list = ET.SubElement(body, "ol")
    for label, detail in CANONICAL_EXECUTION_CYCLE:
        item = ET.SubElement(cycle_list, "li")
        ET.SubElement(item, "strong").text = f"{label}:"
        detail_span = ET.SubElement(item, "span")
        detail_span.text = f" {detail}"

    ET.SubElement(body, "h2").text = "Merge and local-run readiness"
    readiness = ET.SubElement(body, "ul")
    for signal in MERGE_READINESS_SIGNALS:
        ET.SubElement(readiness, "li").text = signal.replace("<artifact_dir>", product_summary["artifact_dir"])

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

    ET.SubElement(body, "h2").text = "Artifact guide"
    guide_table = ET.SubElement(body, "table")
    guide_head = ET.SubElement(guide_table, "tr")
    for heading in ("Order", "Artifact", "Role", "Why open it"):
        ET.SubElement(guide_head, "th").text = heading
    for idx, entry in enumerate(_artifact_entries_present(manifest), start=1):
        tr = ET.SubElement(guide_table, "tr")
        ET.SubElement(tr, "td").text = str(idx)
        td_link = ET.SubElement(tr, "td")
        ET.SubElement(td_link, "a", {"href": entry["name"]}).text = entry["name"]
        ET.SubElement(tr, "td").text = f"{entry['group']} — {entry['label']}"
        ET.SubElement(tr, "td").text = entry["purpose"]

    ET.SubElement(body, "h2").text = "Bundle validation"
    validation_p = ET.SubElement(body, "p")
    validation_p.text = "Re-run validation with "
    ET.SubElement(validation_p, "code").text = f"bnsyn validate-bundle {product_summary['artifact_dir']}"
    validation_p.tail = None

    ET.SubElement(body, "h2").text = "Minimal FAQ"
    faq_list = ET.SubElement(body, "dl")
    for question, answer in FAQ_ENTRIES:
        ET.SubElement(faq_list, "dt").text = question
        ET.SubElement(faq_list, "dd").text = answer

    ET.SubElement(body, "h2").text = "Provenance"
    ET.SubElement(body, "p").text = (
        f"bundle_contract={product_summary['bundle_contract_version']}, "
        f"package_version={product_summary['package_version']}, "
        f"generated_at={_DETERMINISTIC_TIMESTAMP}"
    )
    integrity_table = ET.SubElement(body, "table")
    integrity_head = ET.SubElement(integrity_table, "tr")
    for heading in ("Linked artifact", "SHA-256", "Relationship"):
        ET.SubElement(integrity_head, "th").text = heading
    for artifact_name, digest, relationship in (
        (
            "emergence_plot.png",
            product_summary["primary_visual_sha256"],
            "Primary visual committed in the canonical manifest",
        ),
        (
            product_summary["primary_visual_data_artifact"],
            product_summary["primary_visual_data_sha256"],
            "Underlying numeric trace anchoring the primary visual",
        ),
    ):
        row = ET.SubElement(integrity_table, "tr")
        ET.SubElement(row, "td").text = artifact_name
        ET.SubElement(row, "td").text = digest
        ET.SubElement(row, "td").text = relationship

    ET.indent(root)
    html_string = ET.tostring(root, encoding="unicode", method="html")
    return "<!DOCTYPE html>\n" + html_string + "\n"


def canonical_bundle_contract_supported(contract: str) -> bool:
    return contract == CANONICAL_EXPORT_PROOF_CONTRACT

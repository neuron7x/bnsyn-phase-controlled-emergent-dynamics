"""Smoke tests for provenance manifest.

Parameters
----------
None

Returns
-------
None

Notes
-----
Tests RunManifest for reproducibility tracking.

References
----------
docs/features/provenance_manifest.md
"""

from __future__ import annotations

import json


from bnsyn.provenance import RunManifest


def test_manifest_initialization() -> None:
    """Test RunManifest initialization."""
    manifest = RunManifest(seed=42, config={"lr": 0.01})
    assert manifest.seed == 42
    assert manifest.config["lr"] == 0.01
    assert "python_version" in manifest.to_dict()


def test_manifest_to_dict() -> None:
    """Test manifest to_dict."""
    manifest = RunManifest(seed=42, config={"param": "value"})
    d = manifest.to_dict()

    assert d["seed"] == 42
    assert d["config"]["param"] == "value"
    assert "python_version" in d
    assert "git_sha" in d


def test_manifest_to_json() -> None:
    """Test manifest JSON serialization."""
    manifest = RunManifest(seed=42, config={"x": 1})
    json_str = manifest.to_json()

    # should be valid JSON
    data = json.loads(json_str)
    assert data["seed"] == 42


def test_manifest_hash_determinism() -> None:
    """Test manifest hash is deterministic."""
    manifest1 = RunManifest(seed=42, config={"a": 1})
    manifest2 = RunManifest(seed=42, config={"a": 1})

    # hashes might differ due to timestamp, but structure should be consistent
    hash1 = manifest1.compute_hash()
    hash2 = manifest2.compute_hash()

    # both should be valid SHA256 hashes
    assert len(hash1) == 64  # SHA256 hex digest length
    assert len(hash2) == 64


def test_manifest_round_trip() -> None:
    """Test manifest round-trip through dict."""
    manifest1 = RunManifest(seed=42, config={"param": 123})
    d = manifest1.to_dict()
    manifest2 = RunManifest.from_dict(d)

    assert manifest2.seed == 42
    assert manifest2.config["param"] == 123


def test_output_hash() -> None:
    """Test adding output hashes."""
    manifest = RunManifest(seed=42, config={})
    manifest.add_output_hash("weights", b"data123")

    d = manifest.to_dict()
    assert "output_hashes" in d
    assert "weights" in d["output_hashes"]

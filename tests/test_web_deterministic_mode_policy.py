"""Policy tests for deterministic mode in production."""

from __future__ import annotations

import pytest

from bnsyn.web.config import load_settings


def test_prod_rejects_deterministic_mode_without_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BNSYN_ENVIRONMENT", "prod")
    monkeypatch.setenv("BNSYN_DETERMINISTIC_MODE", "1")
    monkeypatch.setenv("BNSYN_FIXED_NOW", "1700000000")
    monkeypatch.setenv("BNSYN_JWT_SECRET", "0123456789abcdef0123456789abcdef")
    monkeypatch.delenv("BNSYN_ALLOW_DETERMINISTIC_IN_PROD", raising=False)
    with pytest.raises(ValueError, match="forbidden in prod"):
        load_settings()


def test_prod_allows_deterministic_mode_with_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BNSYN_ENVIRONMENT", "prod")
    monkeypatch.setenv("BNSYN_DETERMINISTIC_MODE", "1")
    monkeypatch.setenv("BNSYN_FIXED_NOW", "1700000000")
    monkeypatch.setenv("BNSYN_JWT_SECRET", "0123456789abcdef0123456789abcdef")
    monkeypatch.setenv("BNSYN_ALLOW_DETERMINISTIC_IN_PROD", "1")
    settings = load_settings()
    assert settings.environment == "prod"
    assert settings.deterministic_mode is True

"""Tests for web JWT secret policy enforcement."""

from __future__ import annotations

import pytest

from bnsyn.web.app import create_app
from bnsyn.web.config import Settings, load_settings


def test_dev_mode_rejects_missing_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BNSYN_ENVIRONMENT", "dev")
    monkeypatch.delenv("BNSYN_JWT_SECRET", raising=False)
    with pytest.raises(ValueError, match="BNSYN_JWT_SECRET"):
        load_settings()


def test_dev_mode_rejects_placeholder_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BNSYN_ENVIRONMENT", "dev")
    monkeypatch.setenv("BNSYN_JWT_SECRET", "change-me-in-production")
    with pytest.raises(ValueError, match="blocked placeholder"):
        load_settings()


def test_dev_mode_rejects_short_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BNSYN_ENVIRONMENT", "dev")
    monkeypatch.setenv("BNSYN_JWT_SECRET", "short")
    with pytest.raises(ValueError, match="at least 32"):
        load_settings()


def test_test_mode_allows_test_secret(tmp_path) -> None:
    settings = Settings(
        database_path=str(tmp_path / "test.sqlite3"),
        jwt_secret="test-secret",
        jwt_algorithm="HS256",
        access_token_ttl_seconds=900,
        cookie_secure=False,
        cookie_samesite="lax",
        cookie_name="bnsyn_at",
        csrf_cookie_name="bnsyn_csrf",
        host="127.0.0.1",
        port=8000,
        logical_epoch=1700000000,
        timestamp_value="1970-01-01T00:00:00Z",
        environment="test",
        deterministic_mode=True,
        fixed_now=1700000000,
    )
    app = create_app(settings)
    assert app is not None

"""Coverage-focused unit tests for web support modules."""

from __future__ import annotations

import sqlite3

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from bnsyn.web import cli as web_cli
from bnsyn.web.app import create_app
from bnsyn.web.config import Settings, load_settings
from bnsyn.web.db import connect_db, db_connection, ensure_logical_clock, next_logical_time, next_token_counter
from bnsyn.web.routes_auth import _header_origin


def _settings(tmp_path) -> Settings:
    return Settings(
        database_path=str(tmp_path / "db.sqlite3"),
        jwt_secret="0123456789abcdef0123456789abcdef",
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
        allowed_origins=(),
    )


def test_db_clock_and_counter_paths(tmp_path) -> None:
    conn = connect_db(str(tmp_path / "db.sqlite3"))
    conn.execute("CREATE TABLE IF NOT EXISTS logical_clock (id INTEGER PRIMARY KEY CHECK(id = 1), now INTEGER NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS token_counters (user_id TEXT PRIMARY KEY, counter INTEGER NOT NULL)")
    conn.commit()

    assert ensure_logical_clock(conn, epoch=10) == 10
    assert ensure_logical_clock(conn, epoch=10) == 10
    assert next_logical_time(conn, epoch=10) == 11

    assert next_token_counter(conn, user_id="u1") == 1
    assert next_token_counter(conn, user_id="u1") == 2
    conn.close()


def test_db_connection_context_closes(tmp_path) -> None:
    path = str(tmp_path / "ctx.sqlite3")
    with db_connection(path) as conn:
        conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_load_settings_rejects_invalid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BNSYN_ENVIRONMENT", "bad")
    with pytest.raises(ValueError, match="must be one of"):
        load_settings()


def test_run_web_exit_codes(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(web_cli, "load_settings", lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert web_cli.run_web() == 2

    settings = _settings(tmp_path)
    monkeypatch.setattr(web_cli, "load_settings", lambda: settings)
    monkeypatch.setattr(web_cli.uvicorn, "run", lambda app, host, port: None)
    assert web_cli.run_web() == 0


def test_create_app_and_routes(tmp_path) -> None:
    app = create_app(_settings(tmp_path))
    assert isinstance(app, FastAPI)


def test_header_origin_parsing() -> None:
    assert _header_origin("http://testserver/path") == "http://testserver"
    assert _header_origin("not-a-url") == ""


def test_get_settings_dependency(tmp_path) -> None:
    app = create_app(_settings(tmp_path))
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "app": app}
    request = Request(scope)
    assert request.app.state.settings.cookie_name == "bnsyn_at"

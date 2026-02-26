"""Security tests for web authentication flow."""

from __future__ import annotations

import httpx
import pytest

from bnsyn.web.app import create_app
from bnsyn.web.config import Settings
from bnsyn.web.db import bootstrap_schema, connect_db, insert_tenant, insert_user
from bnsyn.web.security import hash_password


def _seed_user(*, db_path: str, email: str, password: str, role: str = "member", tenant_id: str = "t-1") -> None:
    conn = connect_db(db_path)
    bootstrap_schema(conn)
    insert_tenant(conn, tenant_id=tenant_id, name="Tenant")
    insert_user(
        conn,
        user_id="u-1",
        email=email,
        password_hash=hash_password(password),
        role=role,
        tenant_id=tenant_id,
    )
    conn.commit()
    conn.close()


def _settings(db_path: str, *, ttl_seconds: int = 900, fixed_now: int = 1_700_000_000) -> Settings:
    return Settings(
        database_path=db_path,
        jwt_secret="test-secret",
        jwt_algorithm="HS256",
        access_token_ttl_seconds=ttl_seconds,
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
        fixed_now=fixed_now,
    )


def _client(db_path: str, *, ttl_seconds: int = 900, fixed_now: int = 1_700_000_000) -> httpx.AsyncClient:
    app = create_app(_settings(db_path, ttl_seconds=ttl_seconds, fixed_now=fixed_now))
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.anyio
async def test_login_success_sets_cookie_flags(tmp_path) -> None:
    db_path = str(tmp_path / "test.sqlite3")
    _seed_user(db_path=db_path, email="user@example.com", password="good-password")
    client = _client(db_path)
    try:
        response = await client.post("/token", data={"username": "user@example.com", "password": "good-password"})
    finally:
        await client.aclose()
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert "bnsyn_at" in response.cookies
    assert "bnsyn_csrf" in response.cookies
    set_cookie = "\n".join(response.headers.get_list("set-cookie"))
    assert "bnsyn_at=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "bnsyn_csrf=" in set_cookie


@pytest.mark.anyio
async def test_login_fail_returns_401(tmp_path) -> None:
    db_path = str(tmp_path / "test.sqlite3")
    _seed_user(db_path=db_path, email="user@example.com", password="good-password")
    client = _client(db_path)
    try:
        response = await client.post("/token", data={"username": "user@example.com", "password": "wrong"})
    finally:
        await client.aclose()
    assert response.status_code == 401


@pytest.mark.anyio
async def test_me_requires_auth(tmp_path) -> None:
    db_path = str(tmp_path / "test.sqlite3")
    _seed_user(db_path=db_path, email="user@example.com", password="good-password")
    client = _client(db_path)
    try:
        response = await client.get("/me")
    finally:
        await client.aclose()
    assert response.status_code == 401


@pytest.mark.anyio
async def test_logout_revokes_token(tmp_path) -> None:
    db_path = str(tmp_path / "test.sqlite3")
    _seed_user(db_path=db_path, email="user@example.com", password="good-password")
    client = _client(db_path)
    try:
        login = await client.post("/token", data={"username": "user@example.com", "password": "good-password"})
        assert login.status_code == 200
        before = await client.get("/me")
        assert before.status_code == 200

        forbidden = await client.post("/logout")
        assert forbidden.status_code == 403

        csrf = client.cookies.get("bnsyn_csrf")
        assert csrf is not None
        logout = await client.post("/logout", headers={"x-bnsyn-csrf": csrf})
        assert logout.status_code == 200

        after = await client.get("/me")
    finally:
        await client.aclose()
    assert after.status_code == 401

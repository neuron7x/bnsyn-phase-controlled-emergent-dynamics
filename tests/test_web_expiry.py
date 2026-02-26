"""Tests for web token expiry semantics in deterministic mode."""

from __future__ import annotations

import httpx
import pytest

from bnsyn.web.app import create_app
from bnsyn.web.config import Settings
from bnsyn.web.db import bootstrap_schema, connect_db, insert_tenant, insert_user
from bnsyn.web.security import hash_password


def _seed(db_path: str) -> None:
    conn = connect_db(db_path)
    bootstrap_schema(conn)
    insert_tenant(conn, tenant_id="t-1", name="Tenant")
    insert_user(
        conn,
        user_id="u-1",
        email="user@example.com",
        password_hash=hash_password("pw"),
        role="member",
        tenant_id="t-1",
    )
    conn.commit()
    conn.close()


def _app(db_path: str, fixed_now: int, ttl: int = 2):
    return create_app(
        Settings(
            database_path=db_path,
            jwt_secret="test-secret",
            jwt_algorithm="HS256",
            access_token_ttl_seconds=ttl,
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
    )


@pytest.mark.anyio
async def test_token_expires_in_test_mode(tmp_path) -> None:
    db_path = str(tmp_path / "test.sqlite3")
    _seed(db_path)

    client1 = httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(db_path, fixed_now=100, ttl=2)), base_url="http://testserver")
    try:
        login = await client1.post("/token", data={"username": "user@example.com", "password": "pw"})
        assert login.status_code == 200
        token = client1.cookies.get("bnsyn_at")
        csrf = client1.cookies.get("bnsyn_csrf")
    finally:
        await client1.aclose()

    client2 = httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(db_path, fixed_now=103, ttl=2)), base_url="http://testserver")
    try:
        if token is not None:
            client2.cookies.set("bnsyn_at", token)
        if csrf is not None:
            client2.cookies.set("bnsyn_csrf", csrf)
        me = await client2.get("/me")
    finally:
        await client2.aclose()

    assert me.status_code == 401


@pytest.mark.anyio
async def test_token_not_expired_before_ttl(tmp_path) -> None:
    db_path = str(tmp_path / "test.sqlite3")
    _seed(db_path)

    client1 = httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(db_path, fixed_now=100, ttl=2)), base_url="http://testserver")
    try:
        login = await client1.post("/token", data={"username": "user@example.com", "password": "pw"})
        assert login.status_code == 200
        token = client1.cookies.get("bnsyn_at")
        csrf = client1.cookies.get("bnsyn_csrf")
    finally:
        await client1.aclose()

    client2 = httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(db_path, fixed_now=101, ttl=2)), base_url="http://testserver")
    try:
        if token is not None:
            client2.cookies.set("bnsyn_at", token)
        if csrf is not None:
            client2.cookies.set("bnsyn_csrf", csrf)
        me = await client2.get("/me")
    finally:
        await client2.aclose()

    assert me.status_code == 200

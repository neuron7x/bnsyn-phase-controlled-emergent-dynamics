"""Security tests for RBAC protection."""

from __future__ import annotations

import httpx
import pytest

from bnsyn.web.app import create_app
from bnsyn.web.config import Settings
from bnsyn.web.db import bootstrap_schema, connect_db, insert_tenant, insert_user
from bnsyn.web.security import hash_password


def _seed_user(db_path: str, *, user_id: str, email: str, password: str, role: str) -> None:
    conn = connect_db(db_path)
    bootstrap_schema(conn)
    insert_tenant(conn, tenant_id="tenant-rbac", name="Tenant RBAC")
    insert_user(
        conn,
        user_id=user_id,
        email=email,
        password_hash=hash_password(password),
        role=role,
        tenant_id="tenant-rbac",
    )
    conn.commit()
    conn.close()


def _client(db_path: str) -> httpx.AsyncClient:
    app = create_app(
        Settings(
            database_path=db_path,
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
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


@pytest.mark.anyio
async def test_viewer_cannot_access_admin(tmp_path) -> None:
    db_path = str(tmp_path / "test.sqlite3")
    _seed_user(db_path, user_id="u-viewer", email="viewer@example.com", password="pw", role="viewer")
    client = _client(db_path)
    try:
        login = await client.post("/token", data={"username": "viewer@example.com", "password": "pw"})
        assert login.status_code == 200
        response = await client.get("/admin/ping")
    finally:
        await client.aclose()
    assert response.status_code == 403


@pytest.mark.anyio
async def test_admin_can_access_admin(tmp_path) -> None:
    db_path = str(tmp_path / "test.sqlite3")
    _seed_user(db_path, user_id="u-admin", email="admin@example.com", password="pw", role="admin")
    client = _client(db_path)
    try:
        login = await client.post("/token", data={"username": "admin@example.com", "password": "pw"})
        assert login.status_code == 200
        response = await client.get("/admin/ping")
    finally:
        await client.aclose()
    assert response.status_code == 200

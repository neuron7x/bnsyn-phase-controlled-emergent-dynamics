"""Security tests for tenant context propagation."""

from __future__ import annotations

import httpx
import pytest

from bnsyn.web.app import create_app
from bnsyn.web.config import Settings
from bnsyn.web.db import bootstrap_schema, connect_db, insert_tenant, insert_user
from bnsyn.web.security import hash_password


@pytest.mark.anyio
async def test_me_returns_tenant_id(tmp_path) -> None:
    db_path = str(tmp_path / "test.sqlite3")
    conn = connect_db(db_path)
    bootstrap_schema(conn)
    insert_tenant(conn, tenant_id="tenant-42", name="Tenant 42")
    insert_user(
        conn,
        user_id="u-42",
        email="u42@example.com",
        password_hash=hash_password("pw"),
        role="member",
        tenant_id="tenant-42",
    )
    conn.commit()
    conn.close()

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

    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
    try:
        login = await client.post("/token", data={"username": "u42@example.com", "password": "pw"})
        assert login.status_code == 200
        me = await client.get("/me")
    finally:
        await client.aclose()

    assert me.status_code == 200
    assert me.json()["tenant_id"] == "tenant-42"

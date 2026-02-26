"""Origin/Referer protection tests for logout CSRF flow."""

from __future__ import annotations

import httpx
import pytest

from bnsyn.web.app import create_app
from bnsyn.web.config import Settings
from bnsyn.web.db import bootstrap_schema, connect_db, insert_tenant, insert_user
from bnsyn.web.security import hash_password


@pytest.mark.anyio
async def test_logout_rejects_cross_site_origin_and_allows_same_origin(tmp_path) -> None:
    db_path = str(tmp_path / "test.sqlite3")
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

    app = create_app(
        Settings(
            database_path=db_path,
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
            environment="dev",
            deterministic_mode=False,
            fixed_now=None,
            allowed_origins=("http://testserver",),
        )
    )

    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
    try:
        login = await client.post("/token", data={"username": "user@example.com", "password": "pw"})
        assert login.status_code == 200
        csrf = client.cookies.get("bnsyn_csrf")
        assert csrf is not None

        blocked = await client.post(
            "/logout",
            headers={"x-bnsyn-csrf": csrf, "origin": "https://evil.example"},
        )
        assert blocked.status_code == 403

        allowed = await client.post(
            "/logout",
            headers={"x-bnsyn-csrf": csrf, "origin": "http://testserver"},
        )
        assert allowed.status_code == 200
    finally:
        await client.aclose()

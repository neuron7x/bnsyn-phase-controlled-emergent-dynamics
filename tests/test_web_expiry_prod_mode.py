"""Tests proving dev/prod mode enforces JWT exp via PyJWT."""

from __future__ import annotations

import httpx
import jwt
import pytest

from bnsyn.web.app import create_app
from bnsyn.web.config import Settings
from bnsyn.web.db import bootstrap_schema, connect_db, insert_tenant, insert_user
from bnsyn.web.security import hash_password


@pytest.mark.anyio
async def test_dev_mode_rejects_expired_cookie_token(tmp_path) -> None:
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

    secret = "0123456789abcdef0123456789abcdef"
    app = create_app(
        Settings(
            database_path=db_path,
            jwt_secret=secret,
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

    expired = jwt.encode(
        {
            "sub": "u-1",
            "email": "user@example.com",
            "role": "member",
            "tenant_id": "t-1",
            "jti": "expired-jti-1",
            "iat": 0,
            "exp": 1,
        },
        secret,
        algorithm="HS256",
    )

    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
    try:
        client.cookies.set("bnsyn_at", expired)
        response = await client.get("/me")
    finally:
        await client.aclose()

    assert response.status_code == 401

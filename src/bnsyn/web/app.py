"""FastAPI application factory for bnsyn web perimeter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from bnsyn.web.config import Settings, load_settings
from bnsyn.web.db import bootstrap_schema, connect_db
from bnsyn.web.routes_admin import router as admin_router
from bnsyn.web.routes_auth import router as auth_router
from bnsyn.web.routes_core import router as core_router


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure FastAPI application."""
    resolved_settings = settings or load_settings()
    with connect_db(resolved_settings.database_path) as conn:
        bootstrap_schema(conn)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(title="bnsyn web", lifespan=lifespan)
    app.state.settings = resolved_settings

    app.include_router(core_router)
    app.include_router(auth_router)
    app.include_router(admin_router)

    return app

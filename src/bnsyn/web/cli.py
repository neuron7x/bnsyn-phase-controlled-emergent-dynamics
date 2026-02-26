"""CLI runner utilities for web perimeter."""

from __future__ import annotations

import uvicorn

from bnsyn.web.app import create_app
from bnsyn.web.config import load_settings


def run_web() -> int:
    """Run the bnsyn web ASGI app via uvicorn."""
    try:
        settings = load_settings()
    except ValueError:
        return 2
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    return 0

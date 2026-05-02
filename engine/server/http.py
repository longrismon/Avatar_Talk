"""FastAPI application factory — /ws endpoint, /health, and React SPA static mount."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from .websocket import ConnectionManager
from engine.logging_config import get_logger

if TYPE_CHECKING:
    from engine.orchestrator.state_machine import Orchestrator

log = get_logger("server.http")


def create_app(orchestrator: "Orchestrator", config) -> FastAPI:
    """
    Build and return the FastAPI application.

    Args:
        orchestrator: Running Orchestrator instance.
        config: Settings instance (engine/config.py).
    """
    app = FastAPI(title="Avatar Agent", version="0.1.0")
    manager = ConnectionManager(config.server.auth_token)

    # Wire orchestrator broadcast to WebSocket fan-out
    orchestrator._broadcast = manager.broadcast

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        token = ws.query_params.get("token", "")
        if not await manager.connect(ws, token):
            return
        try:
            await manager.receive_loop(ws, orchestrator)
        finally:
            await manager.disconnect(ws)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "state": orchestrator.state.value}

    # Serve React build if present
    dist = Path(__file__).parents[2] / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="ui")
        log.info("static_ui_mounted", path=str(dist))

    return app

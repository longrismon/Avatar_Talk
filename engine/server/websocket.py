"""WebSocket connection manager — fan-out broadcasts + relay to orchestrator."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

from .auth import verify_token
from engine.logging_config import get_logger

if TYPE_CHECKING:
    from engine.orchestrator.state_machine import Orchestrator

log = get_logger("server.ws")


class ConnectionManager:
    """Manages active WebSocket connections and routes messages to the orchestrator."""

    def __init__(self, auth_token: str) -> None:
        self._auth_token = auth_token
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket, token: str) -> bool:
        """Accept the WebSocket if token is valid. Returns True on success."""
        if not verify_token(token, self._auth_token):
            await ws.close(code=4001, reason="Unauthorized")
            log.warning("ws_auth_failed")
            return False
        await ws.accept()
        self._connections.append(ws)
        log.info("ws_connected", total=len(self._connections))
        return True

    async def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        log.info("ws_disconnected", total=len(self._connections))

    async def broadcast(self, event: dict) -> None:
        """Fan-out a JSON event to all active connections."""
        if not self._connections:
            return
        data = json.dumps(event)
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

    async def receive_loop(self, ws: WebSocket, orchestrator: "Orchestrator") -> None:
        """Read client messages and dispatch events to the orchestrator."""
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning("ws_invalid_json", raw=raw[:120])
                    continue

                msg_type = msg.get("type")

                if msg_type == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

                elif msg_type == "selection":
                    option_id = msg.get("option_id")
                    await orchestrator.handle_event({
                        "type": "response_selected",
                        "data": option_id,
                    })

                elif msg_type == "takeover":
                    await orchestrator.handle_event({
                        "type": "override_action",
                        "data": "end_call",
                    })

                elif msg_type == "resume_ai":
                    await orchestrator.handle_event({
                        "type": "override_action",
                        "data": "resume_ai",
                    })

                else:
                    log.debug("ws_unknown_message", msg_type=msg_type)

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            log.error("ws_receive_error", error=str(exc))

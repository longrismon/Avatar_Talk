"""Tests for Phase 5 — push notification module."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engine.modules.notifications.interface import NotificationClient
from engine.modules.notifications import create_notifier
from engine.orchestrator.state_machine import (
    AgentState,
    Mission,
    Orchestrator,
    StateContext,
)


# ---------------------------------------------------------------------------
# NotificationClient interface contract
# ---------------------------------------------------------------------------


class ConcreteNotifier(NotificationClient):
    """Minimal concrete impl for ABC tests."""

    async def send(self, title: str, body: str, data: dict | None = None) -> None:
        pass

    async def aclose(self) -> None:
        pass


class TestNotificationClientInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            NotificationClient()

    def test_concrete_subclass_instantiates(self):
        n = ConcreteNotifier()
        assert isinstance(n, NotificationClient)

    async def test_send_is_awaitable(self):
        n = ConcreteNotifier()
        await n.send("Title", "Body")

    async def test_aclose_is_awaitable(self):
        n = ConcreteNotifier()
        await n.aclose()


# ---------------------------------------------------------------------------
# FirebaseNotifier unit tests (firebase-admin SDK mocked)
# ---------------------------------------------------------------------------


class TestFirebaseNotifier:
    def _make_notifier(self):
        from engine.modules.notifications.firebase import FirebaseNotifier

        return FirebaseNotifier(
            credentials_path="./creds.json",
            device_token="test-token-abc",
        )

    def test_init_stores_params(self):
        n = self._make_notifier()
        assert n._credentials_path == "./creds.json"
        assert n._device_token == "test-token-abc"
        assert n._app is None

    def test_ensure_app_initializes_once(self):
        import sys

        n = self._make_notifier()
        mock_app = MagicMock()
        mock_fb = MagicMock()
        mock_fb.initialize_app.return_value = mock_app
        mock_fb.credentials.Certificate.return_value = MagicMock()

        with patch.dict(sys.modules, {"firebase_admin": mock_fb, "firebase_admin.credentials": mock_fb.credentials}):
            n._ensure_app()
            n._ensure_app()  # second call should be a no-op
            mock_fb.initialize_app.assert_called_once()
        assert n._app is mock_app

    async def test_send_dispatches_to_executor(self):
        n = self._make_notifier()
        n._send_sync = MagicMock()
        await n.send("Hello", "World", {"key": "val"})
        n._send_sync.assert_called_once_with("Hello", "World", {"key": "val"})

    def test_send_sync_builds_message(self):
        import sys

        n = self._make_notifier()
        n._app = MagicMock()

        mock_messaging = MagicMock()
        mock_messaging.send.return_value = "msg-id-123"

        mock_fb = MagicMock()
        mock_fb.messaging = mock_messaging

        with patch.dict(sys.modules, {"firebase_admin": mock_fb, "firebase_admin.messaging": mock_messaging}):
            n._send_sync("T", "B", {"k": "v"})

        mock_messaging.Notification.assert_called_once_with(title="T", body="B")
        mock_messaging.send.assert_called_once()

    async def test_aclose_deletes_app(self):
        import sys

        n = self._make_notifier()
        saved_app = MagicMock()
        n._app = saved_app

        mock_fb = MagicMock()
        with patch.dict(sys.modules, {"firebase_admin": mock_fb}):
            await n.aclose()
            mock_fb.delete_app.assert_called_once_with(saved_app)

        assert n._app is None

    async def test_aclose_noop_when_no_app(self):
        n = self._make_notifier()
        # Should not raise even if app is None
        await n.aclose()


# ---------------------------------------------------------------------------
# create_notifier factory
# ---------------------------------------------------------------------------


class TestCreateNotifier:
    def test_returns_firebase_notifier(self):
        from engine.modules.notifications.firebase import FirebaseNotifier

        cfg = MagicMock()
        cfg.firebase.credentials_path = "./creds.json"
        cfg.firebase.device_token = "tok"

        notifier = create_notifier(cfg)
        assert isinstance(notifier, FirebaseNotifier)

    def test_device_token_override(self):
        cfg = MagicMock()
        cfg.firebase.credentials_path = "./creds.json"
        cfg.firebase.device_token = "original"

        notifier = create_notifier(cfg, device_token="override")
        assert notifier._device_token == "override"

    def test_falls_back_to_config_token(self):
        cfg = MagicMock()
        cfg.firebase.credentials_path = "./creds.json"
        cfg.firebase.device_token = "from-config"

        notifier = create_notifier(cfg)
        assert notifier._device_token == "from-config"


# ---------------------------------------------------------------------------
# Orchestrator._notify() helper
# ---------------------------------------------------------------------------


class TestOrchestratorNotifyHelper:
    @pytest.fixture
    def broadcast(self):
        events = []

        async def _bc(ev):
            events.append(ev)

        return events, _bc

    def _make_notifier(self):
        n = AsyncMock(spec=NotificationClient)
        return n

    def _make_notifications_config(self, push_enabled=True, events=None):
        cfg = MagicMock()
        cfg.push_enabled = push_enabled
        cfg.events = events or ["review_started", "intervention_needed", "call_connected", "call_ended"]
        return cfg

    async def test_notify_calls_notifier_send(self):
        notifier = self._make_notifier()
        cfg = self._make_notifications_config()
        orch = Orchestrator(notifier=notifier, notifications_config=cfg)

        await orch._notify("review_started", "T", "B", {"k": "v"})

        notifier.send.assert_awaited_once_with(title="T", body="B", data={"k": "v"})

    async def test_notify_noop_when_no_notifier(self):
        orch = Orchestrator()
        # Should not raise
        await orch._notify("review_started", "T", "B")

    async def test_notify_noop_when_push_disabled(self):
        notifier = self._make_notifier()
        cfg = self._make_notifications_config(push_enabled=False)
        orch = Orchestrator(notifier=notifier, notifications_config=cfg)

        await orch._notify("review_started", "T", "B")

        notifier.send.assert_not_awaited()

    async def test_notify_noop_when_event_not_in_list(self):
        notifier = self._make_notifier()
        cfg = self._make_notifications_config(events=["call_ended"])
        orch = Orchestrator(notifier=notifier, notifications_config=cfg)

        await orch._notify("review_started", "T", "B")

        notifier.send.assert_not_awaited()

    async def test_notify_swallows_send_exceptions(self):
        notifier = self._make_notifier()
        notifier.send.side_effect = Exception("FCM unavailable")
        cfg = self._make_notifications_config()
        orch = Orchestrator(notifier=notifier, notifications_config=cfg)

        # Should not propagate
        await orch._notify("review_started", "T", "B")

    async def test_notify_defaults_data_to_empty_dict(self):
        notifier = self._make_notifier()
        cfg = self._make_notifications_config()
        orch = Orchestrator(notifier=notifier, notifications_config=cfg)

        await orch._notify("call_ended", "T", "B")

        _, kwargs = notifier.send.await_args
        assert kwargs["data"] == {}


# ---------------------------------------------------------------------------
# Orchestrator state handler integration tests (Phase 5 notifications)
# ---------------------------------------------------------------------------


class TestNotificationsInStateMachine:
    @pytest.fixture
    def broadcast(self):
        collected = []

        async def _bc(ev):
            collected.append(ev)

        return collected, _bc

    def _notifier(self):
        return AsyncMock(spec=NotificationClient)

    def _notifications_config(self, events=None):
        cfg = MagicMock()
        cfg.push_enabled = True
        cfg.events = events or ["review_started", "intervention_needed", "call_connected", "call_ended"]
        return cfg

    async def test_review_started_sends_notification(self, broadcast):
        collected, bc = broadcast
        notifier = self._notifier()
        cfg = self._notifications_config()

        review_cfg = MagicMock()
        review_cfg.timeout_seconds = 60.0
        review_cfg.auto_select_recommended = True

        orch = Orchestrator(broadcast=bc, notifier=notifier, notifications_config=cfg, review_config=review_cfg)
        orch.state = AgentState.HUMAN_REVIEW
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            response_options=[
                {"id": 1, "text": "Option A", "tone": "direct", "recommended": True},
            ],
        )

        # Pre-buffer a selection so the handler doesn't block
        await orch.handle_event({"type": "response_selected", "data": 1})
        await orch._step()

        notifier.send.assert_awaited_once()
        call_kwargs = notifier.send.await_args.kwargs
        assert call_kwargs["title"] == "Response Review"

    async def test_call_connected_sends_notification(self, broadcast):
        collected, bc = broadcast
        notifier = self._notifier()
        cfg = self._notifications_config()

        orch = Orchestrator(broadcast=bc, notifier=notifier, notifications_config=cfg)
        orch.state = AgentState.AWAITING_CALL

        # Pre-buffer the call_connected event
        await orch.handle_event({"type": "call_connected", "data": None})
        await orch._step()

        notifier.send.assert_awaited_once()
        call_kwargs = notifier.send.await_args.kwargs
        assert "connected" in call_kwargs["title"].lower()

    async def test_call_ended_sends_notification(self, broadcast):
        collected, bc = broadcast
        notifier = self._notifier()
        cfg = self._notifications_config()

        orch = Orchestrator(broadcast=bc, notifier=notifier, notifications_config=cfg)
        orch.state = AgentState.CALL_ENDED
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            turn_number=3,
        )

        await orch._step()

        notifier.send.assert_awaited_once()
        call_kwargs = notifier.send.await_args.kwargs
        assert call_kwargs["data"]["turn_count"] == "3"

    async def test_call_ended_generates_final_summary(self, broadcast):
        collected, bc = broadcast
        notifier = self._notifier()
        cfg = self._notifications_config()

        mock_llm = AsyncMock()
        mock_llm.summarize_call = AsyncMock(return_value={"summary": "Goal was achieved."})

        orch = Orchestrator(
            broadcast=bc,
            llm=mock_llm,
            notifier=notifier,
            notifications_config=cfg,
        )
        orch.state = AgentState.CALL_ENDED
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            transcript=[{"speaker": "other", "text": "Hello"}],
            turn_number=1,
        )

        await orch._step()

        mock_llm.summarize_call.assert_awaited_once()
        # Notification body should contain the summary
        call_kwargs = notifier.send.await_args.kwargs
        assert "Goal was achieved." in call_kwargs["body"]

    async def test_call_ended_notification_skipped_when_disabled(self, broadcast):
        collected, bc = broadcast
        notifier = self._notifier()

        cfg = MagicMock()
        cfg.push_enabled = False
        cfg.events = ["call_ended"]

        orch = Orchestrator(broadcast=bc, notifier=notifier, notifications_config=cfg)
        orch.state = AgentState.CALL_ENDED
        orch.ctx = StateContext(mission=Mission(original_instruction="test"))

        await orch._step()

        notifier.send.assert_not_awaited()

    async def test_intervention_needed_sends_notification(self, broadcast):
        collected, bc = broadcast
        notifier = self._notifier()
        cfg = self._notifications_config()

        mock_result = MagicMock()
        mock_result.status.value = "NEEDS_INTERVENTION"
        from engine.modules.browser.interface import ActionStatus
        mock_result.status = ActionStatus.NEEDS_INTERVENTION
        mock_result.error = "Element not found"
        mock_result.screenshot_path = None

        mock_browser = AsyncMock()
        mock_browser.execute_step = AsyncMock(return_value=mock_result)

        orch = Orchestrator(
            browser=mock_browser,
            broadcast=bc,
            notifier=notifier,
            notifications_config=cfg,
        )
        orch.state = AgentState.BROWSER_ACTION
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            action_plan=[{"action": "click_button", "params": {}}],
            current_step_index=0,
        )

        # Pre-buffer intervention response so handler doesn't block
        await orch.handle_event({"type": "intervention_response", "data": "abort"})
        await orch._step()

        notifier.send.assert_awaited_once()
        call_kwargs = notifier.send.await_args.kwargs
        assert "intervention" in call_kwargs["title"].lower()

"""
Phase 1 integration test — exercises the full flow:
user_instruction → PLANNING → BROWSER_ACTION (n steps) → AWAITING_CALL

All I/O (LLM, browser, logging) is mocked. No real network or browser is used.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from engine.orchestrator.state_machine import AgentState, Mission, Orchestrator, StateContext
from engine.modules.browser.interface import ActionResult, ActionStatus
from engine.modules.llm.interface import ActionPlan


@pytest.fixture
def event_log():
    """Captures all broadcast events."""
    log = []
    async def broadcast(event):
        log.append(event)
    return log, broadcast


@pytest.fixture
def full_plan():
    """A realistic 5-step action plan."""
    return ActionPlan(
        steps=[
            {"action": "open_app", "params": {"app_name": "teams"}},
            {"action": "search_contact", "params": {"name": "Alex"}},
            {"action": "read_chat_history", "params": {"contact": "Alex", "limit": 30}},
            {"action": "start_call", "params": {"contact": "Alex", "video": False}},
            {"action": "grant_permissions", "params": {"mic": True, "camera": False}},
        ],
        mission_summary="Call Alex on Teams to schedule a Sunday meeting.",
        estimated_duration="3-5 minutes",
        conversation_goal="Agree on a time for a Sunday work meeting.",
        success_criteria="Meeting time confirmed by both parties.",
    )


@pytest.fixture
def mock_llm(full_plan):
    llm = AsyncMock()
    llm.generate_plan = AsyncMock(return_value=full_plan)
    return llm


@pytest.fixture
def mock_browser():
    browser = AsyncMock()
    browser.execute_step = AsyncMock(
        return_value=ActionResult(status=ActionStatus.SUCCESS, data={})
    )
    return browser


class TestFullPhase1Flow:
    """Full IDLE → PLANNING → BROWSER_ACTION → AWAITING_CALL flow."""

    async def test_full_flow_succeeds(self, event_log, mock_llm, mock_browser):
        """Complete flow with 5 browser steps all succeeding."""
        collected, broadcast = event_log
        orch = Orchestrator(llm=mock_llm, browser=mock_browser, broadcast=broadcast)

        instruction = "Call Alex on Teams to schedule a Sunday meeting"

        # Feed all needed events up front
        await orch.handle_event({"type": "user_instruction", "data": instruction})
        await orch.handle_event({"type": "plan_response", "data": "confirmed"})

        # IDLE → PLANNING (processes user_instruction)
        assert orch.state == AgentState.IDLE
        await orch._step()
        assert orch.state == AgentState.PLANNING

        # PLANNING → BROWSER_ACTION (processes plan_response "confirmed")
        await orch._step()
        assert orch.state == AgentState.BROWSER_ACTION
        assert len(orch.ctx.action_plan) == 5
        assert orch.ctx.mission.summary == "Call Alex on Teams to schedule a Sunday meeting."

        # BROWSER_ACTION: steps 1-3 (open_app, search_contact, read_chat_history)
        # step 4 is start_call which should trigger AWAITING_CALL
        for _ in range(3):  # open_app, search_contact, read_chat_history
            await orch._step()
        assert orch.state == AgentState.BROWSER_ACTION

        # step 4: start_call → triggers AWAITING_CALL immediately
        await orch._step()
        assert orch.state == AgentState.AWAITING_CALL

    async def test_instruction_is_stored_in_mission(self, event_log, mock_llm, mock_browser):
        """Verifies the instruction flows into the mission context."""
        collected, broadcast = event_log
        orch = Orchestrator(llm=mock_llm, browser=mock_browser, broadcast=broadcast)
        instruction = "Call Alex on Teams"

        await orch.handle_event({"type": "user_instruction", "data": instruction})
        await orch._step()  # IDLE → PLANNING

        assert orch.ctx.mission.original_instruction == instruction

    async def test_plan_steps_stored_correctly(self, event_log, mock_llm, mock_browser):
        """Verifies the LLM plan steps are stored in ctx.action_plan."""
        collected, broadcast = event_log
        orch = Orchestrator(llm=mock_llm, browser=mock_browser, broadcast=broadcast)

        await orch.handle_event({"type": "user_instruction", "data": "Call Alex"})
        await orch.handle_event({"type": "plan_response", "data": "confirmed"})
        await orch._step()  # IDLE → PLANNING
        await orch._step()  # PLANNING → BROWSER_ACTION

        assert len(orch.ctx.action_plan) == 5
        assert orch.ctx.action_plan[0]["action"] == "open_app"
        assert orch.ctx.action_plan[3]["action"] == "start_call"

    async def test_read_chat_history_stores_chat_context(self, event_log, mock_browser):
        """Verifies that read_chat_history step stores messages in ctx.chat_history."""
        collected, broadcast = event_log

        # Plan with only read_chat_history to isolate the test
        mock_llm = AsyncMock()
        mock_llm.generate_plan = AsyncMock(return_value=ActionPlan(
            steps=[
                {"action": "read_chat_history", "params": {"contact": "Alex", "limit": 30}},
            ],
            mission_summary="Read Alex's messages",
            estimated_duration="30s",
        ))

        mock_browser.execute_step = AsyncMock(return_value=ActionResult(
            status=ActionStatus.SUCCESS,
            data={"messages": [
                {"sender": "Alex", "text": "Hey, are you free Sunday?", "timestamp": "10:00"},
                {"sender": "You", "text": "Let me check.", "timestamp": "10:01"},
            ]}
        ))

        orch = Orchestrator(llm=mock_llm, browser=mock_browser, broadcast=broadcast)
        await orch.handle_event({"type": "user_instruction", "data": "Read messages from Alex"})
        await orch.handle_event({"type": "plan_response", "data": "confirmed"})
        await orch._step()  # IDLE → PLANNING
        await orch._step()  # PLANNING → BROWSER_ACTION
        await orch._step()  # BROWSER_ACTION: execute read_chat_history

        assert orch.ctx.chat_history is not None
        assert len(orch.ctx.chat_history) == 2
        assert orch.ctx.chat_history[0]["sender"] == "Alex"

    async def test_broadcast_events_contain_state_changes(self, event_log, mock_llm, mock_browser):
        """Verifies state_changed events are broadcast at each transition."""
        collected, broadcast = event_log
        orch = Orchestrator(llm=mock_llm, browser=mock_browser, broadcast=broadcast)

        await orch.handle_event({"type": "user_instruction", "data": "Call Alex"})
        await orch.handle_event({"type": "plan_response", "data": "confirmed"})
        await orch._step()  # IDLE → PLANNING
        await orch._step()  # PLANNING → BROWSER_ACTION

        state_changes = [e for e in collected if e["type"] == "state_changed"]
        transitions = [(e["from"], e["to"]) for e in state_changes]

        assert ("IDLE", "PLANNING") in transitions
        assert ("PLANNING", "BROWSER_ACTION") in transitions

    async def test_plan_rejected_returns_to_idle(self, event_log, mock_llm, mock_browser):
        """Verifies that rejecting the plan returns to IDLE with clean context."""
        collected, broadcast = event_log
        orch = Orchestrator(llm=mock_llm, browser=mock_browser, broadcast=broadcast)

        await orch.handle_event({"type": "user_instruction", "data": "Call Alex"})
        await orch.handle_event({"type": "plan_response", "data": "rejected"})
        await orch._step()  # IDLE → PLANNING
        await orch._step()  # PLANNING → IDLE (rejected)

        assert orch.state == AgentState.IDLE
        assert orch.ctx.mission is None

    async def test_browser_step_failure_transitions_to_error_with_message(self, event_log, mock_llm):
        """Verifies that a browser step failure produces an ERROR with descriptive message."""
        collected, broadcast = event_log

        mock_browser = AsyncMock()
        mock_browser.execute_step = AsyncMock(return_value=ActionResult(
            status=ActionStatus.FAILED,
            error="Teams sign-in page detected — session expired"
        ))

        orch = Orchestrator(llm=mock_llm, browser=mock_browser, broadcast=broadcast)
        await orch.handle_event({"type": "user_instruction", "data": "Call Alex"})
        await orch.handle_event({"type": "plan_response", "data": "confirmed"})
        await orch.handle_event({"type": "error_response", "data": "abort"})
        await orch._step()  # IDLE → PLANNING
        await orch._step()  # PLANNING → BROWSER_ACTION
        await orch._step()  # BROWSER_ACTION step 1 fails → ERROR

        assert orch.state == AgentState.ERROR
        assert "session expired" in orch.ctx.error_message or "Teams sign-in" in orch.ctx.error_message

        await orch._step()  # ERROR handler runs → broadcasts error event, then → IDLE

        error_events = [e for e in collected if e["type"] == "error"]
        assert len(error_events) >= 1

    async def test_llm_call_uses_correct_instruction(self, event_log, mock_llm, mock_browser):
        """Verifies the LLM is called with the exact instruction from the user."""
        collected, broadcast = event_log
        orch = Orchestrator(llm=mock_llm, browser=mock_browser, broadcast=broadcast)

        instruction = "Call Alex on Teams and schedule a Sunday 2pm meeting"
        await orch.handle_event({"type": "user_instruction", "data": instruction})
        await orch.handle_event({"type": "plan_response", "data": "confirmed"})
        await orch._step()  # IDLE → PLANNING
        await orch._step()  # PLANNING → BROWSER_ACTION

        mock_llm.generate_plan.assert_called_once()
        call_args = mock_llm.generate_plan.call_args
        assert call_args[0][0] == instruction  # first positional arg

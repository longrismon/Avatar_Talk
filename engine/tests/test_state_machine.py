"""Tests for the Orchestrator state machine — Phase 1 states."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from engine.orchestrator.state_machine import AgentState, Orchestrator, StateContext, Mission


@pytest.fixture
def events():
    """Collects broadcast events for assertions."""
    collected = []
    async def broadcast(event):
        collected.append(event)
    return collected, broadcast


@pytest.fixture
def mock_llm():
    from engine.modules.llm.interface import ActionPlan
    llm = AsyncMock()
    llm.generate_plan = AsyncMock(return_value=ActionPlan(
        steps=[
            {"action": "open_app", "params": {"app_name": "teams"}},
            {"action": "search_contact", "params": {"name": "Alex"}},
        ],
        mission_summary="Call Alex on Teams.",
        estimated_duration="3 min",
        conversation_goal="Schedule meeting.",
        success_criteria="Time agreed.",
    ))
    return llm


@pytest.fixture
def mock_browser():
    from engine.modules.browser.interface import ActionResult, ActionStatus
    browser = AsyncMock()
    browser.execute_step = AsyncMock(return_value=ActionResult(status=ActionStatus.SUCCESS))
    return browser


class TestIdleState:
    async def test_starts_in_idle(self, events):
        _, broadcast = events
        orch = Orchestrator(broadcast=broadcast)
        assert orch.state == AgentState.IDLE

    async def test_idle_transitions_to_planning_on_instruction(self, events, mock_llm):
        collected, broadcast = events
        orch = Orchestrator(llm=mock_llm, broadcast=broadcast)

        # Send instruction immediately, then step
        await orch.handle_event({"type": "user_instruction", "data": "Call Alex on Teams"})
        await orch._step()  # processes IDLE → PLANNING

        assert orch.state == AgentState.PLANNING
        assert orch.ctx.mission.original_instruction == "Call Alex on Teams"

    async def test_idle_broadcast_on_entry(self, events):
        collected, broadcast = events
        orch = Orchestrator(broadcast=broadcast)

        await orch.handle_event({"type": "user_instruction", "data": "test"})
        await orch._step()

        types = [e["type"] for e in collected]
        assert "idle" in types or "state_changed" in types


class TestPlanningState:
    async def test_planning_to_browser_action_on_confirmed(self, events, mock_llm, mock_browser):
        collected, broadcast = events
        orch = Orchestrator(llm=mock_llm, browser=mock_browser, broadcast=broadcast)
        orch.state = AgentState.PLANNING
        orch.ctx = StateContext(mission=Mission(original_instruction="Call Alex"))

        # Pre-load the plan_response event
        await orch.handle_event({"type": "plan_response", "data": "confirmed"})
        await orch._step()

        assert orch.state == AgentState.BROWSER_ACTION
        assert len(orch.ctx.action_plan) == 2

    async def test_planning_to_idle_on_rejected(self, events, mock_llm):
        collected, broadcast = events
        orch = Orchestrator(llm=mock_llm, broadcast=broadcast)
        orch.state = AgentState.PLANNING
        orch.ctx = StateContext(mission=Mission(original_instruction="Call Alex"))

        await orch.handle_event({"type": "plan_response", "data": "rejected"})
        await orch._step()

        assert orch.state == AgentState.IDLE

    async def test_planning_to_error_when_no_llm(self, events):
        collected, broadcast = events
        orch = Orchestrator(llm=None, broadcast=broadcast)
        orch.state = AgentState.PLANNING
        orch.ctx = StateContext(mission=Mission(original_instruction="Call Alex"))

        await orch._step()

        assert orch.state == AgentState.ERROR
        assert "No LLM" in orch.ctx.error_message

    async def test_planning_to_error_on_llm_failure(self, events):
        collected, broadcast = events
        mock_llm = AsyncMock()
        mock_llm.generate_plan = AsyncMock(side_effect=Exception("API error"))
        orch = Orchestrator(llm=mock_llm, broadcast=broadcast)
        orch.state = AgentState.PLANNING
        orch.ctx = StateContext(mission=Mission(original_instruction="Call Alex"))

        await orch._step()

        assert orch.state == AgentState.ERROR


class TestBrowserActionState:
    async def test_all_steps_succeed(self, events, mock_browser):
        from engine.modules.browser.interface import ActionResult, ActionStatus
        collected, broadcast = events
        mock_browser.execute_step = AsyncMock(return_value=ActionResult(status=ActionStatus.SUCCESS))

        orch = Orchestrator(browser=mock_browser, broadcast=broadcast)
        orch.state = AgentState.BROWSER_ACTION
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            action_plan=[
                {"action": "open_app", "params": {"app_name": "teams"}},
                {"action": "search_contact", "params": {"name": "Alex"}},
            ],
            current_step_index=0,
        )

        # Execute both steps
        await orch._step()  # step 1: open_app → success, index becomes 1
        assert orch.ctx.current_step_index == 1
        assert orch.state == AgentState.BROWSER_ACTION

        await orch._step()  # step 2: search_contact → success, index becomes 2
        assert orch.ctx.current_step_index == 2
        assert orch.state == AgentState.BROWSER_ACTION

        await orch._step()  # index >= len(plan) → AWAITING_CALL
        assert orch.state == AgentState.AWAITING_CALL

    async def test_step_failure_transitions_to_error(self, events):
        from engine.modules.browser.interface import ActionResult, ActionStatus
        collected, broadcast = events
        mock_browser = AsyncMock()
        mock_browser.execute_step = AsyncMock(
            return_value=ActionResult(status=ActionStatus.FAILED, error="Selector not found")
        )

        orch = Orchestrator(browser=mock_browser, broadcast=broadcast)
        orch.state = AgentState.BROWSER_ACTION
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            action_plan=[{"action": "open_app", "params": {"app_name": "teams"}}],
            current_step_index=0,
        )

        await orch._step()

        assert orch.state == AgentState.ERROR
        assert "Selector not found" in orch.ctx.error_message

    async def test_step_intervention_then_retry(self, events):
        from engine.modules.browser.interface import ActionResult, ActionStatus
        collected, broadcast = events
        mock_browser = AsyncMock()
        mock_browser.execute_step = AsyncMock(
            return_value=ActionResult(
                status=ActionStatus.NEEDS_INTERVENTION,
                error="Login required"
            )
        )

        orch = Orchestrator(browser=mock_browser, broadcast=broadcast)
        orch.state = AgentState.BROWSER_ACTION
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            action_plan=[{"action": "open_app", "params": {}}],
            current_step_index=0,
        )

        # Pre-load intervention response: retry
        await orch.handle_event({"type": "intervention_response", "data": "retry"})
        await orch._step()

        # Should still be in BROWSER_ACTION at the same step
        assert orch.state == AgentState.BROWSER_ACTION
        assert orch.ctx.current_step_index == 0  # not incremented

    async def test_no_browser_transitions_to_error(self, events):
        collected, broadcast = events
        orch = Orchestrator(browser=None, broadcast=broadcast)
        orch.state = AgentState.BROWSER_ACTION
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            action_plan=[{"action": "open_app", "params": {}}],
        )

        await orch._step()
        assert orch.state == AgentState.ERROR


class TestErrorState:
    async def test_error_abort_transitions_to_idle(self, events):
        collected, broadcast = events
        orch = Orchestrator(broadcast=broadcast)
        orch.state = AgentState.ERROR
        orch.ctx = StateContext(
            error_message="Test error",
            error_source_state=AgentState.BROWSER_ACTION,
        )

        await orch.handle_event({"type": "error_response", "data": "abort"})
        await orch._step()

        assert orch.state == AgentState.IDLE

    async def test_error_retry_transitions_to_source_state(self, events):
        collected, broadcast = events
        orch = Orchestrator(broadcast=broadcast)
        orch.state = AgentState.ERROR
        orch.ctx = StateContext(
            error_message="Test error",
            error_source_state=AgentState.PLANNING,
        )

        await orch.handle_event({"type": "error_response", "data": "retry"})
        await orch._step()

        assert orch.state == AgentState.PLANNING


class TestSnapshotAndBroadcast:
    async def test_snapshot_contains_required_fields(self, events):
        collected, broadcast = events
        orch = Orchestrator(broadcast=broadcast)
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test", summary="Test plan"),
            action_plan=[{"action": "open_app", "params": {}}],
            current_step_index=1,
        )
        snap = orch._snapshot()
        assert "state" in snap
        assert "step_index" in snap
        assert "total_steps" in snap
        assert snap["total_steps"] == 1
        assert snap["step_index"] == 1

    async def test_transition_broadcasts_state_changed(self, events):
        collected, broadcast = events
        orch = Orchestrator(broadcast=broadcast)
        await orch._transition(AgentState.PLANNING)

        state_events = [e for e in collected if e["type"] == "state_changed"]
        assert len(state_events) >= 1
        assert state_events[-1]["from"] == "IDLE"
        assert state_events[-1]["to"] == "PLANNING"

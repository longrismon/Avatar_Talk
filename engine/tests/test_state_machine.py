"""Tests for the Orchestrator state machine — Phase 1 states."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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


# ---------------------------------------------------------------------------
# Phase 3 — GENERATING state
# ---------------------------------------------------------------------------

def _make_options() -> list[dict]:
    return [
        {"id": 1, "text": "Option A", "tone": "professional", "recommended": False},
        {"id": 2, "text": "Option B", "tone": "empathetic", "recommended": False},
        {"id": 3, "text": "Option C", "tone": "direct", "recommended": True},
        {"id": 4, "text": "Option D", "tone": "light", "recommended": False},
    ]


@pytest.fixture
def mock_llm_with_responses():
    from engine.modules.llm.interface import ActionPlan, ResponseOptions
    llm = AsyncMock()
    llm.generate_responses = AsyncMock(return_value=ResponseOptions(options=_make_options()))
    llm.summarize_call = AsyncMock(return_value={"summary": "Alex agreed to Monday."})
    return llm


@pytest.fixture
def review_cfg():
    from engine.config import ReviewConfig
    return ReviewConfig(timeout_seconds=5.0, auto_select_recommended=True)


class TestGeneratingState:
    async def test_llm_returns_options_then_transitions_to_human_review(self, events, mock_llm_with_responses, review_cfg):
        collected, broadcast = events
        orch = Orchestrator(llm=mock_llm_with_responses, broadcast=broadcast, review_config=review_cfg)
        orch.state = AgentState.GENERATING
        orch.ctx = StateContext(
            mission=Mission(original_instruction="Call Alex", conversation_goal="Schedule meeting"),
            transcript=[{"speaker": "other", "text": "What time works?", "turn": 0}],
            turn_number=1,
        )

        await orch._step()

        assert orch.state == AgentState.HUMAN_REVIEW
        assert len(orch.ctx.response_options) == 4

    async def test_options_broadcast_emitted(self, events, mock_llm_with_responses, review_cfg):
        collected, broadcast = events
        orch = Orchestrator(llm=mock_llm_with_responses, broadcast=broadcast, review_config=review_cfg)
        orch.state = AgentState.GENERATING
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            transcript=[{"speaker": "other", "text": "Hello", "turn": 0}],
        )

        await orch._step()

        types = [e["type"] for e in collected]
        assert "options" in types

    async def test_no_llm_transitions_to_error(self, events):
        collected, broadcast = events
        orch = Orchestrator(llm=None, broadcast=broadcast)
        orch.state = AgentState.GENERATING
        orch.ctx = StateContext(mission=Mission(original_instruction="test"))

        await orch._step()

        assert orch.state == AgentState.ERROR
        assert "No LLM" in orch.ctx.error_message

    async def test_llm_exception_transitions_to_error(self, events):
        collected, broadcast = events
        mock_llm = AsyncMock()
        mock_llm.generate_responses = AsyncMock(side_effect=Exception("API down"))
        orch = Orchestrator(llm=mock_llm, broadcast=broadcast)
        orch.state = AgentState.GENERATING
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            transcript=[{"speaker": "other", "text": "hi", "turn": 0}],
        )

        await orch._step()

        assert orch.state == AgentState.ERROR
        assert "API down" in orch.ctx.error_message

    async def test_summarization_triggered_after_five_turns(self, events, mock_llm_with_responses, review_cfg):
        collected, broadcast = events
        orch = Orchestrator(llm=mock_llm_with_responses, broadcast=broadcast, review_config=review_cfg)
        orch.state = AgentState.GENERATING
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            transcript=[{"speaker": "other", "text": "hi", "turn": i} for i in range(5)],
            turns_since_summary=4,  # will hit 5 after increment
        )

        # Pre-emptively queue human_review selection so the test doesn't hang
        await orch.handle_event({"type": "response_selected", "data": 1})
        await orch._step()

        # The background task is created; verify summarize_call will eventually be called
        # (give the event loop a tick to run the task)
        await asyncio.sleep(0)
        assert orch.ctx.turns_since_summary == 0  # reset after trigger


# ---------------------------------------------------------------------------
# Phase 3 — HUMAN_REVIEW state
# ---------------------------------------------------------------------------

class TestHumanReviewState:
    async def test_selection_event_sets_response_and_transitions_to_speaking(self, events, review_cfg):
        collected, broadcast = events
        orch = Orchestrator(broadcast=broadcast, review_config=review_cfg)
        orch.state = AgentState.HUMAN_REVIEW
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            response_options=_make_options(),
            turn_number=1,
        )

        await orch.handle_event({"type": "response_selected", "data": 3})
        await orch._step()

        assert orch.state == AgentState.SPEAKING
        assert orch.ctx.selected_response == "Option C"

    async def test_timeout_auto_selects_recommended(self, events):
        collected, broadcast = events
        cfg = MagicMock()
        cfg.timeout_seconds = 0.05
        cfg.auto_select_recommended = True
        orch = Orchestrator(broadcast=broadcast, review_config=cfg)
        orch.state = AgentState.HUMAN_REVIEW
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            response_options=_make_options(),
            turn_number=1,
        )

        await orch._step()

        assert orch.state == AgentState.SPEAKING
        # Option 3 is marked recommended
        assert orch.ctx.selected_response == "Option C"

    async def test_timeout_auto_select_false_picks_first(self, events):
        collected, broadcast = events
        cfg = MagicMock()
        cfg.timeout_seconds = 0.05
        cfg.auto_select_recommended = False
        orch = Orchestrator(broadcast=broadcast, review_config=cfg)
        orch.state = AgentState.HUMAN_REVIEW
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            response_options=_make_options(),
            turn_number=1,
        )

        await orch._step()

        assert orch.state == AgentState.SPEAKING
        assert orch.ctx.selected_response == "Option A"

    async def test_review_started_broadcast_emitted(self, events, review_cfg):
        collected, broadcast = events
        orch = Orchestrator(broadcast=broadcast, review_config=review_cfg)
        orch.state = AgentState.HUMAN_REVIEW
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            response_options=_make_options(),
        )

        # Pre-load selection so it resolves immediately
        await orch.handle_event({"type": "response_selected", "data": 1})
        await orch._step()

        types = [e["type"] for e in collected]
        assert "review_started" in types

    async def test_selection_broadcasts_response_selected(self, events, review_cfg):
        collected, broadcast = events
        orch = Orchestrator(broadcast=broadcast, review_config=review_cfg)
        orch.state = AgentState.HUMAN_REVIEW
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            response_options=_make_options(),
        )

        await orch.handle_event({"type": "response_selected", "data": 2})
        await orch._step()

        sel_events = [e for e in collected if e["type"] == "response_selected"]
        assert len(sel_events) >= 1
        assert sel_events[-1]["text"] == "Option B"
        assert sel_events[-1]["source"] == "human"


# ---------------------------------------------------------------------------
# Phase 4 — SPEAKING state with lip-sync
# ---------------------------------------------------------------------------

class TestSpeakingStateWithLipSync:
    async def _make_tts(self, audio=b"\x00" * 48000):
        tts = MagicMock()

        async def _synth(text):
            yield audio

        tts.synthesize = _synth
        return tts

    async def test_speaking_without_lipsync_transitions_to_listening(self, events):
        collected, broadcast = events
        tts = await self._make_tts()

        mock_sd = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.write = MagicMock()
        mock_sd.RawOutputStream.return_value = mock_stream

        orch = Orchestrator(tts=tts, broadcast=broadcast, virtual_mic_device=None)
        orch.state = AgentState.SPEAKING
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            selected_response="Hello there.",
        )

        await orch._step()

        assert orch.state == AgentState.LISTENING
        assert orch.ctx.selected_response is None

    async def test_speaking_calls_lipsync_generate_video(self, events):
        collected, broadcast = events
        tts = await self._make_tts()

        mock_lipsync = AsyncMock()
        mock_lipsync.generate_video = AsyncMock(return_value=b"\x00\x00\x00\x18ftyp" + b"\x00" * 24)

        mock_lipsync_cfg = MagicMock()
        mock_lipsync_cfg.enabled = True
        mock_lipsync_cfg.reference_image = "./face.png"
        mock_lipsync_cfg.fps = 25

        with patch("engine.modules.lipsync.virtual_camera.inject_video_frames", new=AsyncMock()):
            orch = Orchestrator(
                tts=tts,
                broadcast=broadcast,
                lipsync=mock_lipsync,
                lipsync_config=mock_lipsync_cfg,
                virtual_camera_device="/dev/video10",
            )
            orch.state = AgentState.SPEAKING
            orch.ctx = StateContext(
                mission=Mission(original_instruction="test"),
                selected_response="Hi.",
            )

            await orch._step()

        mock_lipsync.generate_video.assert_awaited_once()
        assert orch.state == AgentState.LISTENING

    async def test_speaking_lipsync_failure_does_not_crash(self, events):
        collected, broadcast = events
        tts = await self._make_tts()

        mock_lipsync = AsyncMock()
        mock_lipsync.generate_video = AsyncMock(side_effect=Exception("GPU OOM"))

        mock_lipsync_cfg = MagicMock()
        mock_lipsync_cfg.enabled = True
        mock_lipsync_cfg.reference_image = "./face.png"
        mock_lipsync_cfg.fps = 25

        orch = Orchestrator(
            tts=tts,
            broadcast=broadcast,
            lipsync=mock_lipsync,
            lipsync_config=mock_lipsync_cfg,
            virtual_camera_device="/dev/video10",
        )
        orch.state = AgentState.SPEAKING
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            selected_response="Hello.",
        )

        await orch._step()

        # Despite lipsync failure, should still reach LISTENING
        assert orch.state == AgentState.LISTENING

    async def test_speaking_lipsync_disabled_skips_video(self, events):
        collected, broadcast = events
        tts = await self._make_tts()

        mock_lipsync = AsyncMock()

        mock_lipsync_cfg = MagicMock()
        mock_lipsync_cfg.enabled = False  # disabled

        orch = Orchestrator(
            tts=tts,
            broadcast=broadcast,
            lipsync=mock_lipsync,
            lipsync_config=mock_lipsync_cfg,
        )
        orch.state = AgentState.SPEAKING
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            selected_response="Hello.",
        )

        await orch._step()

        mock_lipsync.generate_video.assert_not_awaited()
        assert orch.state == AgentState.LISTENING

    async def test_speaking_broadcasts_speaking_and_complete(self, events):
        collected, broadcast = events
        tts = await self._make_tts()

        orch = Orchestrator(tts=tts, broadcast=broadcast)
        orch.state = AgentState.SPEAKING
        orch.ctx = StateContext(
            mission=Mission(original_instruction="test"),
            selected_response="Test message.",
        )

        await orch._step()

        types = [e["type"] for e in collected]
        assert "speaking" in types
        assert "speaking_complete" in types


class TestSpeakingStateBargeIn:
    """Barge-in detection during SPEAKING state."""

    def _make_vad_result(self, is_speech=True, probability=0.9):
        from engine.modules.audio.vad import VADResult
        return VADResult(is_speech=is_speech, probability=probability, utterance_complete=False)

    def _make_tts(self):
        tts = MagicMock()

        async def _synth(text):
            # One 100ms chunk of silence (3200 bytes at 16kHz/16-bit)
            yield b"\x00" * 3200

        tts.synthesize = _synth
        return tts

    def _make_vad_cfg(self, barge_in_enabled=True, threshold=0.3):
        cfg = MagicMock()
        cfg.barge_in_enabled = barge_in_enabled
        cfg.barge_in_threshold = threshold
        return cfg

    def _make_orch(self, barge_in_enabled=True, broadcast=None):
        mock_vad = MagicMock()
        mock_vad.process_chunk.return_value = []  # no speech by default

        orch = Orchestrator(
            tts=self._make_tts(),
            vad=mock_vad,
            vad_config=self._make_vad_cfg(barge_in_enabled=barge_in_enabled),
            broadcast=broadcast,
        )
        orch.state = AgentState.SPEAKING
        orch.ctx = StateContext(
            mission=Mission(original_instruction="x"),
            selected_response="Hello there.",
        )
        return orch, mock_vad

    async def test_no_barge_in_transitions_to_listening(self):
        orch, _ = self._make_orch()
        await orch._step()
        assert orch.state == AgentState.LISTENING

    async def test_barge_in_detected_transitions_to_listening(self):
        orch, mock_vad = self._make_orch()
        mock_vad.process_chunk.return_value = [self._make_vad_result()]
        await orch.handle_event({"type": "audio_chunk", "data": b"\x00" * 320})
        await orch._step()
        assert orch.state == AgentState.LISTENING

    async def test_barge_in_broadcasts_barge_in_event(self, events):
        collected, broadcast = events
        orch, mock_vad = self._make_orch(broadcast=broadcast)
        mock_vad.process_chunk.return_value = [self._make_vad_result()]
        await orch.handle_event({"type": "audio_chunk", "data": b"\x00" * 320})
        await orch._step()
        types = [e["type"] for e in collected]
        assert "barge_in" in types

    async def test_barge_in_skips_speculative_generation(self):
        mock_llm = AsyncMock()
        orch, mock_vad = self._make_orch()
        orch._llm = mock_llm
        mock_vad.process_chunk.return_value = [self._make_vad_result()]
        await orch.handle_event({"type": "audio_chunk", "data": b"\x00" * 320})
        await orch._step()
        assert orch._speculative_task is None

    async def test_barge_in_disabled_completes_normally(self, events):
        collected, broadcast = events
        orch, mock_vad = self._make_orch(barge_in_enabled=False, broadcast=broadcast)
        mock_vad.process_chunk.return_value = [self._make_vad_result()]
        await orch.handle_event({"type": "audio_chunk", "data": b"\x00" * 320})
        await orch._step()
        assert orch.state == AgentState.LISTENING
        types = [e["type"] for e in collected]
        assert "barge_in" not in types

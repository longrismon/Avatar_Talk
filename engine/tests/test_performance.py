"""Tests for Phase 6 — PerformanceTracker."""
import time

import pytest

from engine.modules.performance import PerformanceTracker, TurnMetrics


class TestTurnMetrics:
    def _make(self, **kwargs) -> TurnMetrics:
        return TurnMetrics(call_id="abc", turn_number=1, **kwargs)

    def test_llm_latency_zero_when_incomplete(self):
        m = self._make()
        assert m.llm_latency_ms == 0

    def test_llm_latency_computed(self):
        m = self._make(llm_start_ms=1000, llm_complete_ms=1300)
        assert m.llm_latency_ms == 300

    def test_tts_latency_computed(self):
        m = self._make(llm_complete_ms=1300, tts_first_chunk_ms=1450)
        assert m.tts_latency_ms == 150

    def test_end_to_end_latency(self):
        m = self._make(utterance_end_ms=1000, tts_first_chunk_ms=1800)
        assert m.end_to_end_latency_ms == 800

    def test_total_turn_ms(self):
        m = self._make(utterance_end_ms=1000, speaking_complete_ms=3500)
        assert m.total_turn_ms == 2500

    def test_zero_when_timestamps_missing(self):
        m = self._make()
        assert m.end_to_end_latency_ms == 0
        assert m.total_turn_ms == 0


class TestPerformanceTracker:
    def test_start_turn_returns_metrics(self):
        tracker = PerformanceTracker()
        m = tracker.start_turn("call1", 1)
        assert isinstance(m, TurnMetrics)
        assert m.call_id == "call1"
        assert m.turn_number == 1

    def test_record_sets_timestamp(self):
        tracker = PerformanceTracker()
        tracker.start_turn("c", 1)
        before = int(time.time() * 1000)
        tracker.record("utterance_end")
        after = int(time.time() * 1000)
        assert before <= tracker._current.utterance_end_ms <= after

    def test_record_noop_when_no_current_turn(self):
        tracker = PerformanceTracker()
        tracker.record("utterance_end")  # should not raise

    def test_record_ignores_unknown_stage(self):
        tracker = PerformanceTracker()
        tracker.start_turn("c", 1)
        tracker.record("unknown_stage")  # should not raise or set anything
        assert tracker._current.utterance_end_ms == 0

    def test_get_percentiles_empty(self):
        tracker = PerformanceTracker()
        result = tracker.get_percentiles("end_to_end")
        assert result == {"p50": 0, "p95": 0, "p99": 0, "count": 0}

    def test_get_percentiles_single_turn(self):
        tracker = PerformanceTracker()
        m = tracker.start_turn("c", 1)
        m.utterance_end_ms = 1000
        m.tts_first_chunk_ms = 1800
        result = tracker.get_percentiles("end_to_end")
        assert result["p50"] == 800
        assert result["count"] == 1

    def test_get_percentiles_multiple_turns(self):
        tracker = PerformanceTracker()
        base_ms = 1_700_000_000_000  # realistic epoch ms
        latencies = [600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]
        for i, lat in enumerate(latencies):
            m = tracker.start_turn("c", i)
            m.utterance_end_ms = base_ms
            m.tts_first_chunk_ms = base_ms + lat
        result = tracker.get_percentiles("end_to_end")
        assert result["count"] == 10
        assert result["p50"] > 0

    def test_summary_contains_all_metrics(self):
        tracker = PerformanceTracker()
        s = tracker.summary()
        assert "end_to_end" in s
        assert "llm" in s
        assert "tts" in s
        assert "total_turn" in s
        assert s["turns_recorded"] == 0

    def test_summary_turns_recorded(self):
        tracker = PerformanceTracker()
        for i in range(3):
            tracker.start_turn("c", i)
        assert tracker.summary()["turns_recorded"] == 3

    def test_multiple_start_turn_calls(self):
        tracker = PerformanceTracker()
        tracker.start_turn("c", 1)
        tracker.start_turn("c", 2)
        assert len(tracker._turns) == 2
        assert tracker._current.turn_number == 2


class TestOrchestratorPerfIntegration:
    """Verify PerformanceTracker is wired into the orchestrator."""

    def test_orchestrator_has_perf_tracker(self):
        from engine.orchestrator.state_machine import Orchestrator
        orch = Orchestrator()
        from engine.modules.performance import PerformanceTracker
        assert isinstance(orch.perf, PerformanceTracker)

    def test_call_id_in_state_context(self):
        from engine.orchestrator.state_machine import StateContext
        ctx = StateContext()
        assert ctx.call_id == ""


class TestConversationPhase:
    def _orch(self, turn: int):
        from engine.orchestrator.state_machine import Orchestrator, StateContext, Mission
        orch = Orchestrator()
        orch.ctx = StateContext(mission=Mission(original_instruction="x"), turn_number=turn)
        return orch

    def test_early_phase(self):
        assert self._orch(0)._conversation_phase() == "early"
        assert self._orch(4)._conversation_phase() == "early"

    def test_mid_phase(self):
        assert self._orch(5)._conversation_phase() == "mid"
        assert self._orch(14)._conversation_phase() == "mid"

    def test_late_phase(self):
        assert self._orch(15)._conversation_phase() == "late"
        assert self._orch(100)._conversation_phase() == "late"

    def test_adaptive_timeout_early_is_longer(self):
        orch_early = self._orch(0)
        orch_late = self._orch(20)
        assert orch_early._adaptive_review_timeout(5.0) > orch_late._adaptive_review_timeout(5.0)

    def test_adaptive_timeout_mid_equals_base(self):
        orch = self._orch(7)
        assert orch._adaptive_review_timeout(6.0) == pytest.approx(6.0)

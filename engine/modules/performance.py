"""Per-turn latency tracking for the Avatar Agent pipeline.

Records timestamps at each stage of the LISTENING→GENERATING→SPEAKING loop
and exposes percentile summaries for the performance dashboard.
"""
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional


STAGES = ("utterance_end", "llm_start", "llm_complete", "tts_first_chunk", "speaking_complete")


@dataclass
class TurnMetrics:
    call_id: str
    turn_number: int
    utterance_end_ms: int = 0
    llm_start_ms: int = 0
    llm_complete_ms: int = 0
    tts_first_chunk_ms: int = 0
    speaking_complete_ms: int = 0

    @property
    def llm_latency_ms(self) -> int:
        if self.llm_complete_ms and self.llm_start_ms:
            return self.llm_complete_ms - self.llm_start_ms
        return 0

    @property
    def tts_latency_ms(self) -> int:
        if self.tts_first_chunk_ms and self.llm_complete_ms:
            return self.tts_first_chunk_ms - self.llm_complete_ms
        return 0

    @property
    def end_to_end_latency_ms(self) -> int:
        """Utterance-end → first audible response byte."""
        if self.tts_first_chunk_ms and self.utterance_end_ms:
            return self.tts_first_chunk_ms - self.utterance_end_ms
        return 0

    @property
    def total_turn_ms(self) -> int:
        """Utterance-end → speaking complete."""
        if self.speaking_complete_ms and self.utterance_end_ms:
            return self.speaking_complete_ms - self.utterance_end_ms
        return 0


class PerformanceTracker:
    """Records per-stage timestamps for each turn and reports latency percentiles.

    Usage in orchestrator:
        self._perf.record("utterance_end")
        # ... LLM call ...
        self._perf.record("llm_complete")

    Stages (in order): utterance_end, llm_start, llm_complete, tts_first_chunk, speaking_complete
    """

    def __init__(self) -> None:
        self._turns: list[TurnMetrics] = []
        self._current: Optional[TurnMetrics] = None

    def start_turn(self, call_id: str, turn_number: int) -> TurnMetrics:
        m = TurnMetrics(call_id=call_id, turn_number=turn_number)
        self._turns.append(m)
        self._current = m
        return m

    def record(self, stage: str) -> None:
        """Record the current wall-clock time for the given pipeline stage."""
        if self._current is None or stage not in STAGES:
            return
        setattr(self._current, f"{stage}_ms", int(time.time() * 1000))

    def _percentile(self, values: list[int], pct: float) -> int:
        if not values:
            return 0
        idx = max(0, int(len(values) * pct) - 1)
        return sorted(values)[idx]

    def get_percentiles(self, metric: str = "end_to_end") -> dict:
        """Return p50/p95/p99 for the given metric across all recorded turns.

        metric: "end_to_end", "llm", "tts", "total_turn"
        """
        attr = f"{metric}_latency_ms" if metric != "total_turn" else "total_turn_ms"
        values = [getattr(t, attr) for t in self._turns if getattr(t, attr, 0) > 0]
        if not values:
            return {"p50": 0, "p95": 0, "p99": 0, "count": 0}
        return {
            "p50": int(statistics.median(values)),
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99),
            "count": len(values),
        }

    def summary(self) -> dict:
        return {
            "end_to_end": self.get_percentiles("end_to_end"),
            "llm": self.get_percentiles("llm"),
            "tts": self.get_percentiles("tts"),
            "total_turn": self.get_percentiles("total_turn"),
            "turns_recorded": len(self._turns),
        }

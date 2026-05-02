"""Structured event schema for cross-module logging (ADR-002)."""
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class LogEvent:
    """Standardized log event written to call_{call_id}.jsonl.

    Every module that emits timed or auditable events should use this
    schema so that session replay and post-call debugging work correctly.
    """
    event_type: str
    module: str                          # "stt", "tts", "browser", "orchestrator", "ui"
    call_id: str = ""
    turn_number: int = 0
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    duration_ms: Optional[int] = None
    payload: dict = field(default_factory=dict)
    level: str = "info"


# ---------------------------------------------------------------------------
# Event type constants — used in Orchestrator.handle_event() / _wait_for_event()
# ---------------------------------------------------------------------------

# Inbound from UI / caller
EV_USER_INSTRUCTION = "user_instruction"
EV_PLAN_RESPONSE = "plan_response"          # "confirmed" | "rejected"
EV_INTERVENTION_RESPONSE = "intervention_response"  # "retry" | "skip" | "abort"
EV_ERROR_RESPONSE = "error_response"        # "retry" | "abort"
EV_CALL_CONNECTED = "call_connected"
EV_OVERRIDE_ACTION = "override_action"      # "resume_ai" | "end_call"
EV_AUDIO_CHUNK = "audio_chunk"              # bytes — raw PCM from capture device
EV_RESPONSE_SELECTED = "response_selected"  # int — option index (Phase 3)

# Outbound broadcasts (orchestrator → UI)
EV_STATE_CHANGED = "state_changed"
EV_PLAN_READY = "plan_ready"
EV_BROWSER_STEP = "browser_step"
EV_INTERVENTION_NEEDED = "intervention_needed"
EV_PARTIAL_TRANSCRIPT = "partial_transcript"
EV_UTTERANCE_COMPLETE = "utterance_complete"
EV_SPEAKING = "speaking"
EV_SPEAKING_COMPLETE = "speaking_complete"
EV_CALL_ENDED = "call_ended"
EV_ERROR = "error"
EV_IDLE = "idle"

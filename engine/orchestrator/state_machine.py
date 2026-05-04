"""
Orchestrator — central state machine for the Avatar Agent.

Phase 1 implements: IDLE, PLANNING, BROWSER_ACTION, CALL_ENDED, ERROR.
All other states have stub handlers that log and await further implementation.
"""
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

from engine.logging_config import get_logger
from engine.modules.browser.interface import ActionStatus

log = get_logger("orchestrator")


class AgentState(Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    BROWSER_ACTION = "BROWSER_ACTION"
    AWAITING_CALL = "AWAITING_CALL"
    LISTENING = "LISTENING"
    GENERATING = "GENERATING"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    SPEAKING = "SPEAKING"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    CALL_ENDED = "CALL_ENDED"
    ERROR = "ERROR"


@dataclass
class Mission:
    original_instruction: str
    summary: str = ""
    conversation_goal: str = ""
    success_criteria: str = ""


@dataclass
class StateContext:
    mission: Optional[Mission] = None
    action_plan: list[dict] = field(default_factory=list)
    current_step_index: int = 0
    transcript: list[dict] = field(default_factory=list)
    chat_history: Optional[str] = None
    turn_number: int = 0
    call_start_time: Optional[float] = None
    error_source_state: Optional[AgentState] = None
    error_message: Optional[str] = None
    selected_response: Optional[str] = None
    # Phase 3 fields
    call_summary_so_far: str = ""
    response_options: list[dict] = field(default_factory=list)
    turns_since_summary: int = 0
    # Phase 6 fields
    call_id: str = ""


class Orchestrator:
    """
    Central state machine for the Avatar Agent.

    The orchestrator receives events from the UI and from internal modules,
    transitions between states, and executes actions in each state.

    Usage (Phase 1, CLI mode):
        orchestrator = Orchestrator(llm=llm_client, browser=browser_pool)
        await orchestrator.handle_event({"type": "user_instruction", "data": "Call Alex"})
        await orchestrator.run_until_idle()
    """

    def __init__(
        self,
        llm=None,
        browser=None,
        stt=None,
        tts=None,
        vad=None,
        virtual_mic_device: Optional[str] = None,
        broadcast: Optional[Callable] = None,
        review_config=None,
        principal_profile: Optional[dict] = None,
        lipsync=None,
        lipsync_config=None,
        virtual_camera_device: Optional[str] = None,
        notifier=None,
        notifications_config=None,
        vad_config=None,
    ):
        """
        Args:
            llm: LLMClient instance (for planning)
            browser: BrowserPool instance (for browser automation)
            stt: STTClient instance (Phase 2+)
            tts: TTSClient instance (Phase 2+)
            vad: SileroVAD instance (Phase 2+)
            virtual_mic_device: sounddevice device name for TTS injection (Phase 2+)
            broadcast: Async callable that receives event dicts to send to connected UIs.
                       Defaults to a no-op if not provided.
            review_config: ReviewConfig from engine/config.py (Phase 3+)
            principal_profile: dict loaded from profiles/default.json (Phase 3+)
            lipsync: LipSyncClient instance (Phase 4+)
            lipsync_config: LipSyncConfig from engine/config.py (Phase 4+)
            virtual_camera_device: v4l2loopback device path for video injection (Phase 4+)
            notifier: NotificationClient instance (Phase 5+)
            notifications_config: NotificationsConfig from engine/config.py (Phase 5+)
            vad_config: VADConfig from engine/config.py (Phase 7+, barge-in detection)
        """
        self.state = AgentState.IDLE
        self.ctx = StateContext()
        self._llm = llm
        self._browser = browser
        self._stt = stt
        self._tts = tts
        self._vad = vad
        self._vad_config = vad_config
        self._virtual_mic_device = virtual_mic_device
        self._broadcast = broadcast or self._default_broadcast
        self._review_config = review_config
        self._principal_profile = principal_profile or {}
        self._lipsync = lipsync
        self._lipsync_config = lipsync_config
        self._virtual_camera_device = virtual_camera_device
        self._notifier = notifier
        self._notifications_config = notifications_config

        # Phase 6: per-call performance tracking
        from engine.modules.performance import PerformanceTracker
        self._perf = PerformanceTracker()

        # Phase 6: speculative LLM pre-generation task (started at end of SPEAKING)
        self._speculative_task: Optional[asyncio.Task] = None

        # Event queue: UI events are pushed here, state handlers drain it
        self._event_queue: asyncio.Queue = asyncio.Queue()

        # Pending event futures: state handlers wait on specific event types
        self._pending_events: dict[str, asyncio.Future] = {}

        # Buffer for events that arrived before _wait_for_event was called
        self._event_buffer: dict[str, list] = {}

    @staticmethod
    async def _default_broadcast(event: dict) -> None:
        """Default broadcast: log the event."""
        log.debug("broadcast", event_type=event.get("type"), data=event)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def handle_event(self, event: dict) -> None:
        """Receive an event from the UI or an internal module.

        Events with a matching pending future resolve that future.
        All events are also pushed to the queue for state handlers that poll.
        Events with no current waiter are buffered so _wait_for_event can
        consume them even if they arrived before the wait was registered.
        """
        event_type = event.get("type")
        data = event.get("data")

        # Resolve pending waiter if any
        future = self._pending_events.pop(event_type, None)
        if future and not future.done():
            future.set_result(data)
        else:
            # No waiter yet — buffer the data so _wait_for_event can pick it up
            if event_type not in self._event_buffer:
                self._event_buffer[event_type] = []
            self._event_buffer[event_type].append(data)

        await self._event_queue.put(event)

    async def run_until_idle(self, timeout: float = 300.0) -> None:
        """Run the state machine until it returns to IDLE or ERROR.

        Used in CLI mode. In server mode, call run() instead.

        Args:
            timeout: Max seconds to run before giving up.
        """
        deadline = time.monotonic() + timeout
        while self.state not in (AgentState.IDLE, AgentState.ERROR):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.error("run_timeout", state=self.state.value)
                break
            await asyncio.wait_for(self._step(), timeout=min(remaining, 60.0))

    async def run(self) -> None:
        """Main loop: run state handlers continuously until stopped."""
        while True:
            try:
                await self._step()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("unhandled_error", state=self.state.value, error=str(e))
                self.ctx.error_message = str(e)
                self.ctx.error_source_state = self.state
                await self._transition(AgentState.ERROR)

    # -------------------------------------------------------------------------
    # Internal: state machine step
    # -------------------------------------------------------------------------

    async def _step(self) -> None:
        """Execute the current state's handler once."""
        handlers = {
            AgentState.IDLE: self._handle_idle,
            AgentState.PLANNING: self._handle_planning,
            AgentState.BROWSER_ACTION: self._handle_browser_action,
            AgentState.AWAITING_CALL: self._handle_awaiting_call,
            AgentState.LISTENING: self._handle_listening,
            AgentState.GENERATING: self._handle_generating,
            AgentState.HUMAN_REVIEW: self._handle_human_review,
            AgentState.SPEAKING: self._handle_speaking,
            AgentState.MANUAL_OVERRIDE: self._handle_manual_override,
            AgentState.CALL_ENDED: self._handle_call_ended,
            AgentState.ERROR: self._handle_error,
        }
        handler = handlers[self.state]
        await handler()

    async def _transition(self, new_state: AgentState, **meta) -> None:
        """Transition to a new state and broadcast the change."""
        old_state = self.state
        self.state = new_state
        log.info(
            "state_transition",
            from_state=old_state.value,
            to_state=new_state.value,
            step=self.ctx.current_step_index,
            turn=self.ctx.turn_number,
            call_id=self.ctx.call_id,
            **meta,
        )
        await self._broadcast({
            "type": "state_changed",
            "from": old_state.value,
            "to": new_state.value,
            "context": self._snapshot(),
        })

    def _snapshot(self) -> dict:
        """Return a serializable snapshot of the current context."""
        return {
            "state": self.state.value,
            "turn_number": self.ctx.turn_number,
            "step_index": self.ctx.current_step_index,
            "total_steps": len(self.ctx.action_plan),
            "error": self.ctx.error_message,
            "mission_summary": self.ctx.mission.summary if self.ctx.mission else None,
        }

    async def _wait_for_event(self, event_type: str, timeout: float = 60.0) -> Any:
        """Wait for a specific event type from the UI.

        If the event already arrived (buffered by handle_event before this call),
        return it immediately without waiting.

        Args:
            event_type: The event type string to wait for.
            timeout: Seconds to wait before raising asyncio.TimeoutError.

        Returns:
            The event data payload.
        """
        # Check if an event of this type arrived before we started waiting
        buffer = self._event_buffer.get(event_type)
        if buffer:
            return buffer.pop(0)

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_events[event_type] = future
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_events.pop(event_type, None)
            raise

    async def _notify(
        self,
        event_type: str,
        title: str,
        body: str,
        data: dict | None = None,
    ) -> None:
        """Send a push notification if enabled and the event type is subscribed."""
        if not self._notifier:
            return
        cfg = self._notifications_config
        if not cfg or not cfg.push_enabled:
            return
        if event_type not in (cfg.events or []):
            return
        try:
            await self._notifier.send(title=title, body=body, data=data or {})
        except Exception as exc:
            log.warning("notification_failed", notification_event=event_type, error=str(exc))

    # -------------------------------------------------------------------------
    # Phase 6 helpers
    # -------------------------------------------------------------------------

    def _conversation_phase(self) -> str:
        """Classify where we are in the call for adaptive timeout tuning.

        Returns "early" (turns 0-4), "mid" (5-14), or "late" (15+).
        """
        t = self.ctx.turn_number
        if t < 5:
            return "early"
        if t < 15:
            return "mid"
        return "late"

    def _adaptive_review_timeout(self, base_timeout: float) -> float:
        """Scale the review timeout based on conversation phase.

        Early calls need more time (principal is still calibrating); late calls
        are faster because the principal has built a rhythm.
        """
        multipliers = {"early": 1.5, "mid": 1.0, "late": 0.75}
        return base_timeout * multipliers[self._conversation_phase()]

    @property
    def perf(self):
        """Public accessor for the PerformanceTracker (for testing and dashboard)."""
        return self._perf

    # -------------------------------------------------------------------------
    # State handlers — Phase 1 (fully implemented)
    # -------------------------------------------------------------------------

    async def _handle_idle(self) -> None:
        """IDLE: wait for a user_instruction event."""
        log.info("state_idle", message="Waiting for instruction")
        await self._broadcast({"type": "idle", "message": "Ready for instructions"})

        instruction = await self._wait_for_event("user_instruction", timeout=86400.0)
        call_id = uuid.uuid4().hex[:8]
        log.info("instruction_received", instruction=instruction, call_id=call_id)

        self.ctx = StateContext(
            mission=Mission(original_instruction=instruction),
            call_id=call_id,
            call_start_time=time.monotonic(),
        )
        await self._transition(AgentState.PLANNING)

    async def _handle_planning(self) -> None:
        """PLANNING: call LLM to generate action plan, ask user to confirm."""
        if not self._llm:
            self.ctx.error_message = "No LLM client configured"
            self.ctx.error_source_state = AgentState.PLANNING
            await self._transition(AgentState.ERROR)
            return

        log.info("planning_start", instruction=self.ctx.mission.original_instruction)
        await self._broadcast({"type": "planning", "message": "Generating action plan..."})

        try:
            from engine.modules.browser.registry import list_supported_apps
            available_apps = list_supported_apps()

            plan = await asyncio.wait_for(
                self._llm.generate_plan(
                    self.ctx.mission.original_instruction,
                    available_apps,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            self.ctx.error_message = "LLM planning timed out after 30 seconds"
            self.ctx.error_source_state = AgentState.PLANNING
            await self._transition(AgentState.ERROR)
            return
        except Exception as e:
            self.ctx.error_message = f"LLM planning failed: {e}"
            self.ctx.error_source_state = AgentState.PLANNING
            await self._transition(AgentState.ERROR)
            return

        self.ctx.action_plan = plan.steps
        self.ctx.mission.summary = plan.mission_summary
        self.ctx.mission.conversation_goal = plan.conversation_goal
        self.ctx.mission.success_criteria = plan.success_criteria
        self.ctx.current_step_index = 0

        await self._broadcast({
            "type": "plan_ready",
            "plan": {
                "steps": plan.steps,
                "mission_summary": plan.mission_summary,
                "estimated_duration": plan.estimated_duration,
                "conversation_goal": plan.conversation_goal,
                "success_criteria": plan.success_criteria,
            },
        })
        log.info("plan_ready", step_count=len(plan.steps), summary=plan.mission_summary)

        try:
            confirmation = await self._wait_for_event("plan_response", timeout=120.0)
        except asyncio.TimeoutError:
            log.info("plan_timeout", message="No response within 120s, aborting")
            await self._transition(AgentState.IDLE)
            return

        if confirmation == "confirmed":
            await self._transition(AgentState.BROWSER_ACTION)
        else:
            log.info("plan_rejected")
            self.ctx = StateContext()
            await self._transition(AgentState.IDLE)

    async def _handle_browser_action(self) -> None:
        """BROWSER_ACTION: execute the next step in the action plan."""
        if not self._browser:
            self.ctx.error_message = "No browser pool configured"
            self.ctx.error_source_state = AgentState.BROWSER_ACTION
            await self._transition(AgentState.ERROR)
            return

        if self.ctx.current_step_index >= len(self.ctx.action_plan):
            # All steps done
            log.info("browser_action_complete", total_steps=len(self.ctx.action_plan))
            await self._transition(AgentState.AWAITING_CALL)
            return

        step = self.ctx.action_plan[self.ctx.current_step_index]
        step_num = self.ctx.current_step_index + 1
        total = len(self.ctx.action_plan)

        log.info(
            "browser_step_start",
            step=step_num,
            total=total,
            action=step["action"],
            params=step.get("params", {}),
        )
        await self._broadcast({
            "type": "browser_step",
            "step": step_num,
            "total": total,
            "action": step["action"],
            "params": step.get("params", {}),
            "status": "running",
        })

        result = await self._browser.execute_step(step)

        if result.status == ActionStatus.SUCCESS:
            # If this step returned chat history, store it
            if step["action"] == "read_chat_history" and result.data:
                self.ctx.chat_history = result.data.get("messages")
                log.info("chat_history_loaded", message_count=len(self.ctx.chat_history or []))

            self.ctx.current_step_index += 1

            await self._broadcast({
                "type": "browser_step",
                "step": step_num,
                "total": total,
                "action": step["action"],
                "status": "success",
            })
            log.info("browser_step_success", step=step_num, action=step["action"])

            # If we just started the call, move to AWAITING_CALL immediately
            if step["action"] == "start_call":
                await self._transition(AgentState.AWAITING_CALL)
            else:
                # Stay in BROWSER_ACTION (next step will be processed on next _step())
                pass  # state stays BROWSER_ACTION

        elif result.status == ActionStatus.NEEDS_INTERVENTION:
            log.warning(
                "browser_step_intervention",
                step=step_num,
                action=step["action"],
                error=result.error,
            )
            await self._broadcast({
                "type": "intervention_needed",
                "step": step_num,
                "message": result.error,
                "screenshot": result.screenshot_path,
            })
            await self._notify(
                "intervention_needed",
                "Manual Intervention Required",
                f"Step {step_num} ({step['action']}) needs attention",
                {"step": str(step_num), "action": step["action"]},
            )

            # Wait for user response (retry/skip/abort)
            try:
                response = await self._wait_for_event("intervention_response", timeout=300.0)
            except asyncio.TimeoutError:
                log.info("intervention_timeout", message="No response within 300s, aborting")
                self.ctx.error_message = "Intervention timed out"
                self.ctx.error_source_state = AgentState.BROWSER_ACTION
                await self._transition(AgentState.ERROR)
                return

            if response == "retry":
                # Stay in BROWSER_ACTION, same step index
                pass
            elif response == "skip":
                self.ctx.current_step_index += 1
                # Stay in BROWSER_ACTION
            else:  # abort or unknown
                self.ctx = StateContext()
                await self._transition(AgentState.IDLE)

        else:  # FAILED
            log.error(
                "browser_step_failed",
                step=step_num,
                action=step["action"],
                error=result.error,
            )
            await self._broadcast({
                "type": "browser_step",
                "step": step_num,
                "total": total,
                "action": step["action"],
                "status": "failed",
                "error": result.error,
            })
            self.ctx.error_message = result.error or f"Step {step_num} ({step['action']}) failed"
            self.ctx.error_source_state = AgentState.BROWSER_ACTION
            await self._transition(AgentState.ERROR)

    # -------------------------------------------------------------------------
    # State handlers — Phase 1 stubs (Phase 2+ to implement)
    # -------------------------------------------------------------------------

    async def _handle_awaiting_call(self) -> None:
        """AWAITING_CALL: stub — waits for a manual event to continue."""
        log.info("state_awaiting_call", message="[Phase 2] Waiting for call to connect")
        await self._broadcast({"type": "awaiting_call"})
        # In Phase 2 this will monitor the browser for call connection.
        # For now, wait for a manual event or transition to CALL_ENDED.
        try:
            event = await self._wait_for_event("call_connected", timeout=60.0)
            await self._notify(
                "call_connected",
                "Call Connected",
                "The call has been connected. AI avatar is now active.",
            )
            await self._transition(AgentState.LISTENING)
        except asyncio.TimeoutError:
            log.info("awaiting_call_timeout")
            await self._transition(AgentState.CALL_ENDED)

    async def _handle_listening(self) -> None:
        if not self._stt or not self._vad:
            log.info("state_listening", message="[Phase 2] STT/VAD not configured")
            await self._transition(AgentState.CALL_ENDED)
            return

        log.info("state_listening")
        await self._broadcast({"type": "listening"})

        utterance_buffer = bytearray()

        try:
            while True:
                audio_data = await self._wait_for_event("audio_chunk", timeout=30.0)
                if not audio_data:
                    continue

                chunk = audio_data if isinstance(audio_data, (bytes, bytearray)) else bytes(audio_data)
                utterance_buffer.extend(chunk)

                # Partial transcript for live UI display
                partial = await self._stt.transcribe_chunk(chunk)
                if partial.text:
                    await self._broadcast({"type": "partial_transcript", "text": partial.text})

                # VAD: detect utterance boundary
                for vad_result in self._vad.process_chunk(chunk):
                    if not vad_result.utterance_complete:
                        continue

                    final = await self._stt.transcribe_utterance(bytes(utterance_buffer))
                    utterance_buffer = bytearray()
                    self._vad.reset()

                    if not final.text.strip():
                        continue

                    turn = {
                        "speaker": "other",
                        "text": final.text,
                        "turn": self.ctx.turn_number,
                    }
                    self.ctx.transcript.append(turn)
                    self.ctx.turn_number += 1
                    self._perf.start_turn(self.ctx.call_id, self.ctx.turn_number)
                    self._perf.record("utterance_end")
                    log.info(
                        "utterance_received",
                        turn=self.ctx.turn_number,
                        call_id=self.ctx.call_id,
                        preview=final.text[:80],
                    )
                    await self._broadcast({
                        "type": "utterance_complete",
                        "text": final.text,
                        "turn": self.ctx.turn_number,
                    })
                    await self._transition(AgentState.GENERATING)
                    return

        except asyncio.TimeoutError:
            log.info("listening_timeout", message="No audio for 30 s — ending call")
            await self._transition(AgentState.CALL_ENDED)
        except Exception as exc:
            self.ctx.error_message = f"Listening failed: {exc}"
            self.ctx.error_source_state = AgentState.LISTENING
            await self._transition(AgentState.ERROR)

    async def _handle_generating(self) -> None:
        if not self._llm:
            self.ctx.error_message = "No LLM configured"
            self.ctx.error_source_state = AgentState.GENERATING
            await self._transition(AgentState.ERROR)
            return

        await self._broadcast({"type": "generating"})
        log.info("state_generating", turn=self.ctx.turn_number, call_id=self.ctx.call_id)

        recent_turns = self.ctx.transcript[-8:]
        context_payload = {
            "principal_profile": self._principal_profile,
            "mission_goal": self.ctx.mission.conversation_goal if self.ctx.mission else "",
            "success_criteria": self.ctx.mission.success_criteria if self.ctx.mission else "",
            "call_summary_so_far": self.ctx.call_summary_so_far,
            "recent_turns": recent_turns,
            "current_utterance": recent_turns[-1]["text"] if recent_turns else "",
        }

        # Check if a speculative result is already available (started at end of previous SPEAKING)
        options_result = None
        if self._speculative_task is not None and self._speculative_task.done():
            try:
                options_result = self._speculative_task.result()
                log.info("speculative_result_used", turn=self.ctx.turn_number, call_id=self.ctx.call_id)
            except Exception:
                options_result = None
            self._speculative_task = None

        if options_result is None:
            self._perf.record("llm_start")
            try:
                options_result = await asyncio.wait_for(
                    self._llm.generate_responses(context_payload),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                self.ctx.error_message = "Response generation timed out"
                self.ctx.error_source_state = AgentState.GENERATING
                await self._transition(AgentState.ERROR)
                return
            except Exception as exc:
                self.ctx.error_message = f"Response generation failed: {exc}"
                self.ctx.error_source_state = AgentState.GENERATING
                await self._transition(AgentState.ERROR)
                return
            self._perf.record("llm_complete")

        self.ctx.response_options = options_result.options

        # Background summarization every 5 turns
        self.ctx.turns_since_summary += 1
        if self.ctx.turns_since_summary >= 5:
            asyncio.create_task(self._run_summarization())
            self.ctx.turns_since_summary = 0

        await self._broadcast({
            "type": "options",
            "options": self.ctx.response_options,
            "turn": self.ctx.turn_number,
        })
        await self._transition(AgentState.HUMAN_REVIEW)

    async def _run_summarization(self) -> None:
        """Background task: update call_summary_so_far via LLM."""
        try:
            mission_dict = {
                "conversation_goal": self.ctx.mission.conversation_goal if self.ctx.mission else "",
                "summary": self.ctx.call_summary_so_far,
            }
            result = await asyncio.wait_for(
                self._llm.summarize_call(self.ctx.transcript, mission_dict),
                timeout=10.0,
            )
            self.ctx.call_summary_so_far = result.get("summary", "")
            log.info("summarization_complete", preview=self.ctx.call_summary_so_far[:80])
        except Exception as exc:
            log.warning("summarization_failed", error=str(exc))

    async def _run_speculative_generation(self):
        """Background task: start LLM response generation speculatively.

        Called immediately after SPEAKING completes, before the other party
        has finished their next utterance.  Uses the current transcript context
        (no current_utterance) so the result may be stale, but in most turns
        the conversational direction is predictable from context.

        The result is consumed in _handle_generating if the task is done by
        the time the utterance ends — giving effectively zero LLM wait time.
        """
        try:
            recent_turns = self.ctx.transcript[-8:]
            context_payload = {
                "principal_profile": self._principal_profile,
                "mission_goal": self.ctx.mission.conversation_goal if self.ctx.mission else "",
                "success_criteria": self.ctx.mission.success_criteria if self.ctx.mission else "",
                "call_summary_so_far": self.ctx.call_summary_so_far,
                "recent_turns": recent_turns,
                "current_utterance": "",  # speculative — utterance not yet known
            }
            return await asyncio.wait_for(
                self._llm.generate_responses(context_payload),
                timeout=15.0,
            )
        except Exception as exc:
            log.debug("speculative_generation_failed", error=str(exc))
            return None

    async def _handle_human_review(self) -> None:
        base_timeout = self._review_config.timeout_seconds if self._review_config else 5.0
        timeout = self._adaptive_review_timeout(base_timeout)
        auto_select = self._review_config.auto_select_recommended if self._review_config else True

        await self._broadcast({
            "type": "review_started",
            "options": self.ctx.response_options,
            "timeout_seconds": timeout,
            "turn": self.ctx.turn_number,
        })
        await self._notify(
            "review_started",
            "Response Review",
            f"Select a response for turn {self.ctx.turn_number} ({timeout}s to auto-select)",
            {"turn": str(self.ctx.turn_number), "timeout": str(int(timeout))},
        )
        log.info("state_human_review", timeout=timeout)

        try:
            selected_id = await self._wait_for_event("response_selected", timeout=timeout)
            option = next(
                (o for o in self.ctx.response_options if o["id"] == selected_id), None
            )
            selected_text = option["text"] if option else self.ctx.response_options[0]["text"]
            selection_source = "human"
        except asyncio.TimeoutError:
            if auto_select:
                rec = next(
                    (o for o in self.ctx.response_options if o.get("recommended")),
                    self.ctx.response_options[0] if self.ctx.response_options else None,
                )
                selected_text = rec["text"] if rec else ""
                selection_source = "auto"
            else:
                selected_text = self.ctx.response_options[0]["text"] if self.ctx.response_options else ""
                selection_source = "auto_first"

        self.ctx.selected_response = selected_text
        log.info(
            "response_selected",
            source=selection_source,
            turn=self.ctx.turn_number,
            preview=selected_text[:60],
        )
        await self._broadcast({
            "type": "response_selected",
            "text": selected_text,
            "source": selection_source,
        })
        await self._transition(AgentState.SPEAKING)

    async def _handle_speaking(self) -> None:
        text = self.ctx.selected_response
        if not self._tts or not text:
            log.info("state_speaking", message="TTS or response not ready")
            await self._transition(AgentState.LISTENING)
            return

        log.info("state_speaking", preview=text[:80])
        await self._broadcast({"type": "speaking", "text": text})

        try:
            # Collect full audio first (needed for lip-sync; audio is short enough to buffer)
            audio_chunks: list[bytes] = []
            first_chunk = True
            async for chunk in self._tts.synthesize(text):
                audio_chunks.append(chunk)
                if first_chunk:
                    self._perf.record("tts_first_chunk")
                    first_chunk = False
            full_audio = b"".join(audio_chunks)

            # Run audio injection with optional barge-in monitoring and lip-sync
            barge_in_event = asyncio.Event()
            barge_in_enabled = (
                self._vad is not None
                and self._vad_config is not None
                and self._vad_config.barge_in_enabled
            )

            inject_task = asyncio.create_task(
                self._inject_audio_bytes(full_audio, barge_in_event)
            )
            lipsync_task = None
            if self._lipsync and self._lipsync_config and self._lipsync_config.enabled:
                lipsync_task = asyncio.create_task(self._inject_lipsync(full_audio))
            monitor_task = None
            if barge_in_enabled:
                monitor_task = asyncio.create_task(self._monitor_barge_in(barge_in_event))

            # Wait for injection to finish (returns early if barge_in_event is set)
            await asyncio.gather(inject_task, return_exceptions=True)
            barged_in = barge_in_event.is_set()

            # Cancel monitor — no longer needed regardless of outcome
            if monitor_task:
                monitor_task.cancel()
                await asyncio.gather(monitor_task, return_exceptions=True)

            # On barge-in cancel lipsync immediately; otherwise wait for it to finish
            if lipsync_task:
                if barged_in:
                    lipsync_task.cancel()
                result = await asyncio.gather(lipsync_task, return_exceptions=True)
                if not barged_in and isinstance(result[0], Exception):
                    log.warning("speaking_subtask_error", error=str(result[0]))

            self.ctx.selected_response = None
            self._perf.record("speaking_complete")
            log.info(
                "speaking_complete",
                call_id=self.ctx.call_id,
                barge_in=barged_in,
                e2e_ms=self._perf._current.end_to_end_latency_ms if self._perf._current else 0,
            )
            await self._broadcast({"type": "speaking_complete"})

            if barged_in:
                await self._broadcast({"type": "barge_in"})
                log.info("barge_in_transition", call_id=self.ctx.call_id)
            elif self._llm:
                # Kick off speculative LLM pre-generation for the next expected response.
                # Skipped on barge-in because the incoming utterance will update context.
                self._speculative_task = asyncio.create_task(
                    self._run_speculative_generation()
                )

            await self._transition(AgentState.LISTENING)

        except Exception as exc:
            self.ctx.error_message = f"Speaking failed: {exc}"
            self.ctx.error_source_state = AgentState.SPEAKING
            await self._transition(AgentState.ERROR)

    async def _inject_audio_bytes(
        self, audio: bytes, stop_event: Optional[asyncio.Event] = None
    ) -> None:
        """Inject collected TTS audio bytes to the virtual mic device.

        Chunks audio into 100ms segments and checks stop_event between chunks
        so barge-in can abort playback mid-sentence.
        """
        # Yield once so concurrent tasks (e.g. barge-in monitor) can start
        # before injection begins consuming chunks.
        await asyncio.sleep(0)

        from engine.modules.audio.virtual_devices import inject_audio

        # 100ms at 16 kHz / 16-bit PCM = 3200 bytes per chunk
        CHUNK_SIZE = 16000 * 2 * 100 // 1000

        async def _chunked():
            offset = 0
            while offset < len(audio):
                if stop_event and stop_event.is_set():
                    return
                yield audio[offset : offset + CHUNK_SIZE]
                offset += CHUNK_SIZE

        if self._virtual_mic_device:
            await inject_audio(_chunked(), device=self._virtual_mic_device)

    async def _monitor_barge_in(self, stop_event: asyncio.Event) -> None:
        """Background task: drain audio_chunk events through VAD; sets stop_event on speech onset."""
        if not self._vad:
            return
        threshold = (
            self._vad_config.barge_in_threshold if self._vad_config else 0.3
        )
        try:
            while not stop_event.is_set():
                try:
                    audio_data = await self._wait_for_event("audio_chunk", timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if not audio_data:
                    continue
                chunk = (
                    audio_data
                    if isinstance(audio_data, (bytes, bytearray))
                    else bytes(audio_data)
                )
                for result in self._vad.process_chunk(chunk):
                    if result.is_speech and result.probability >= threshold:
                        log.info(
                            "barge_in_detected",
                            call_id=self.ctx.call_id,
                            probability=result.probability,
                        )
                        stop_event.set()
                        return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("barge_in_monitor_error", error=str(exc))

    async def _inject_lipsync(self, audio: bytes) -> None:
        """Generate lip-sync video from audio and push frames to virtual camera."""
        from engine.modules.lipsync.virtual_camera import inject_video_frames

        video_bytes = await asyncio.wait_for(
            self._lipsync.generate_video(
                audio_pcm=audio,
                reference_image=self._lipsync_config.reference_image,
            ),
            timeout=30.0,
        )
        if video_bytes and self._virtual_camera_device:
            await inject_video_frames(
                video_bytes,
                device=self._virtual_camera_device,
                fps=self._lipsync_config.fps,
            )

    async def _handle_manual_override(self) -> None:
        log.info("state_manual_override", message="Manual override active — waiting for resume or end")
        await self._broadcast({"type": "manual_override_active"})
        try:
            event = await self._wait_for_event("override_action", timeout=3600.0)
            if event == "resume_ai":
                await self._transition(AgentState.LISTENING)
            else:
                await self._transition(AgentState.CALL_ENDED)
        except asyncio.TimeoutError:
            await self._transition(AgentState.CALL_ENDED)

    async def _handle_call_ended(self) -> None:
        """CALL_ENDED: generate final summary, send notification, reset to IDLE."""
        log.info("state_call_ended", turn_count=self.ctx.turn_number)

        # Generate final call summary if LLM and transcript are available
        final_summary = self.ctx.call_summary_so_far
        if self._llm and self.ctx.transcript:
            try:
                mission_dict = {
                    "conversation_goal": self.ctx.mission.conversation_goal if self.ctx.mission else "",
                    "summary": self.ctx.call_summary_so_far,
                }
                result = await asyncio.wait_for(
                    self._llm.summarize_call(self.ctx.transcript, mission_dict),
                    timeout=10.0,
                )
                final_summary = result.get("summary", self.ctx.call_summary_so_far)
                log.info("final_summary_complete", preview=final_summary[:80])
            except Exception as exc:
                log.warning("final_summary_failed", error=str(exc))

        await self._broadcast({
            "type": "call_ended",
            "turn_count": self.ctx.turn_number,
            "mission_summary": self.ctx.mission.summary if self.ctx.mission else None,
            "final_summary": final_summary,
        })
        await self._notify(
            "call_ended",
            "Call Ended",
            final_summary or f"Call completed after {self.ctx.turn_number} turns",
            {"turn_count": str(self.ctx.turn_number)},
        )

        self.ctx = StateContext()
        await self._transition(AgentState.IDLE)

    async def _handle_error(self) -> None:
        """ERROR: broadcast error, wait for retry or abort."""
        log.error(
            "state_error",
            message=self.ctx.error_message,
            source_state=self.ctx.error_source_state.value if self.ctx.error_source_state else None,
        )
        await self._broadcast({
            "type": "error",
            "message": self.ctx.error_message,
            "source_state": self.ctx.error_source_state.value if self.ctx.error_source_state else None,
        })

        try:
            response = await self._wait_for_event("error_response", timeout=300.0)
        except asyncio.TimeoutError:
            response = "abort"

        if response == "retry" and self.ctx.error_source_state:
            error_state = self.ctx.error_source_state
            self.ctx.error_message = None
            self.ctx.error_source_state = None
            await self._transition(error_state)
        else:
            self.ctx = StateContext()
            await self._transition(AgentState.IDLE)

"""
Orchestrator — central state machine for the Avatar Agent.

Phase 1 implements: IDLE, PLANNING, BROWSER_ACTION, CALL_ENDED, ERROR.
All other states have stub handlers that log and await further implementation.
"""
import asyncio
import time
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
        """
        self.state = AgentState.IDLE
        self.ctx = StateContext()
        self._llm = llm
        self._browser = browser
        self._stt = stt
        self._tts = tts
        self._vad = vad
        self._virtual_mic_device = virtual_mic_device
        self._broadcast = broadcast or self._default_broadcast
        self._review_config = review_config
        self._principal_profile = principal_profile or {}
        self._lipsync = lipsync
        self._lipsync_config = lipsync_config
        self._virtual_camera_device = virtual_camera_device

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

    # -------------------------------------------------------------------------
    # State handlers — Phase 1 (fully implemented)
    # -------------------------------------------------------------------------

    async def _handle_idle(self) -> None:
        """IDLE: wait for a user_instruction event."""
        log.info("state_idle", message="Waiting for instruction")
        await self._broadcast({"type": "idle", "message": "Ready for instructions"})

        instruction = await self._wait_for_event("user_instruction", timeout=86400.0)
        log.info("instruction_received", instruction=instruction)

        self.ctx = StateContext(mission=Mission(original_instruction=instruction))
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
                    log.info(
                        "utterance_received",
                        turn=self.ctx.turn_number,
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
        log.info("state_generating", turn=self.ctx.turn_number)

        recent_turns = self.ctx.transcript[-8:]
        context_payload = {
            "principal_profile": self._principal_profile,
            "mission_goal": self.ctx.mission.conversation_goal if self.ctx.mission else "",
            "success_criteria": self.ctx.mission.success_criteria if self.ctx.mission else "",
            "call_summary_so_far": self.ctx.call_summary_so_far,
            "recent_turns": recent_turns,
            "current_utterance": recent_turns[-1]["text"] if recent_turns else "",
        }

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

    async def _handle_human_review(self) -> None:
        timeout = self._review_config.timeout_seconds if self._review_config else 5.0
        auto_select = self._review_config.auto_select_recommended if self._review_config else True

        await self._broadcast({
            "type": "review_started",
            "options": self.ctx.response_options,
            "timeout_seconds": timeout,
            "turn": self.ctx.turn_number,
        })
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
            async for chunk in self._tts.synthesize(text):
                audio_chunks.append(chunk)
            full_audio = b"".join(audio_chunks)

            # Run audio injection and optional lip-sync video in parallel
            tasks = [self._inject_audio_bytes(full_audio)]
            if self._lipsync and self._lipsync_config and self._lipsync_config.enabled:
                tasks.append(self._inject_lipsync(full_audio))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    log.warning("speaking_subtask_error", error=str(r))

            self.ctx.selected_response = None
            log.info("speaking_complete")
            await self._broadcast({"type": "speaking_complete"})
            await self._transition(AgentState.LISTENING)

        except Exception as exc:
            self.ctx.error_message = f"Speaking failed: {exc}"
            self.ctx.error_source_state = AgentState.SPEAKING
            await self._transition(AgentState.ERROR)

    async def _inject_audio_bytes(self, audio: bytes) -> None:
        """Inject collected TTS audio bytes to the virtual mic device."""
        from engine.modules.audio.virtual_devices import inject_audio

        async def _single_chunk():
            yield audio

        if self._virtual_mic_device:
            await inject_audio(_single_chunk(), device=self._virtual_mic_device)

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
        """CALL_ENDED: log summary and reset to IDLE."""
        log.info("state_call_ended", turn_count=self.ctx.turn_number)
        await self._broadcast({
            "type": "call_ended",
            "turn_count": self.ctx.turn_number,
            "mission_summary": self.ctx.mission.summary if self.ctx.mission else None,
        })
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

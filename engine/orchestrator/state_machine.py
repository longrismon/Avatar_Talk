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

    def __init__(self, llm=None, browser=None, broadcast: Optional[Callable] = None):
        """
        Args:
            llm: LLMClient instance (for planning)
            browser: BrowserPool instance (for browser automation)
            broadcast: Async callable that receives event dicts to send to connected UIs.
                       Defaults to a no-op if not provided.
        """
        self.state = AgentState.IDLE
        self.ctx = StateContext()
        self._llm = llm
        self._browser = browser
        self._broadcast = broadcast or self._default_broadcast

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
        log.info("state_listening", message="[Phase 2] STT pipeline not yet implemented")
        await self._broadcast({"type": "listening"})
        await self._transition(AgentState.CALL_ENDED)

    async def _handle_generating(self) -> None:
        log.info("state_generating", message="[Phase 3] LLM response generation not yet implemented")
        await self._transition(AgentState.CALL_ENDED)

    async def _handle_human_review(self) -> None:
        log.info("state_human_review", message="[Phase 3] HITL not yet implemented")
        await self._transition(AgentState.CALL_ENDED)

    async def _handle_speaking(self) -> None:
        log.info("state_speaking", message="[Phase 4] TTS not yet implemented")
        await self._transition(AgentState.LISTENING)

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

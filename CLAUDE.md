# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from the repo root unless noted. The Python package lives in `engine/`.

```bash
# Install (from engine/ directory)
cd engine
pip install -e ".[dev]"
playwright install chromium

# Run all tests
cd engine && pytest tests/ -v

# Run a single test file
cd engine && pytest tests/test_state_machine.py -v

# Run a single test by name
cd engine && pytest tests/test_state_machine.py::test_idle_transitions_to_planning -v

# CLI commands (from repo root)
export ANTHROPIC_API_KEY="your-key"
python avatar.py plan "Call Alex on Teams to schedule a meeting" --dry-run
python avatar.py health
python avatar.py version

# Override config path
AVATAR_CONFIG=engine/config.yaml python avatar.py health
```

Optional dependency groups: `.[stt]` (faster-whisper), `.[tts]` (sounddevice), `.[ml]` (torch).

## Architecture

### Phase status

**Phase 1 is fully implemented.** Phases 2–6 are stubs. The state machine, browser pool, and LLM planning are all working. STT/TTS, HITL review, the HTTP/WebSocket server, and lip-sync are not.

### Data flow (Phase 1)

```
avatar.py CLI  →  LLMClient.generate_plan()  →  ActionPlan (list of steps)
                                                       ↓
                                              Orchestrator (state machine)
                                                       ↓
                                              BrowserPool.execute_step()
                                                       ↓
                                              TeamsAutomation (Playwright)
```

### Orchestrator / state machine (`engine/orchestrator/state_machine.py`)

The `Orchestrator` class is the central controller. It holds an `AgentState` enum and a `StateContext` dataclass. The full state flow is:

`IDLE → PLANNING → BROWSER_ACTION → AWAITING_CALL → LISTENING → GENERATING → HUMAN_REVIEW → SPEAKING → CALL_ENDED`

Phase 1 fully implements: `IDLE`, `PLANNING`, `BROWSER_ACTION`, `CALL_ENDED`, `ERROR`.
All others are stubs that immediately transition to `CALL_ENDED`.

State handlers communicate with callers via `_broadcast()` (sends event dicts to connected UIs) and `_wait_for_event()` (awaits a specific event type from the UI). The CLI wires these up directly; in server mode the WebSocket will drive them.

### Configuration (`engine/config.py`)

Pydantic v2 models loaded from `engine/config.yaml`. All `${ENV_VAR}` references in the YAML are interpolated at load time. Access via `load_config(path)` (returns a `Settings` instance) or `get_config()` (returns the cached singleton). The `AVATAR_CONFIG` env var overrides the default path.

### Browser module (`engine/modules/browser/`)

- `interface.py` — abstract `BrowserAutomation` base class that all platform implementations must satisfy
- `registry.py` — `APP_REGISTRY` maps app name strings to loader functions; `get_automation_class("teams")` returns `TeamsAutomation`. Slack/Discord are registered but disabled (commented out).
- `pool.py` — `BrowserPool` manages a single persistent Playwright Chromium context. `execute_step(step_dict)` dispatches by calling `getattr(automation, step["action"])(**step["params"])`, so action names in the LLM plan must exactly match method names on `BrowserAutomation`.
- `teams.py` — `TeamsAutomation`: uses a **three-tier selector strategy** — each UI element has a list of selectors tried in order (`aria` label → `data-tid` → role/placeholder). This makes the automation resilient to Teams UI changes.

### LLM module (`engine/modules/llm/`)

- `interface.py` — abstract `LLMClient` with `generate_plan()`, `generate_responses()` (Phase 3), `summarize_call()` (Phase 5)
- `client.py` — `AnthropicLLM` (raw `httpx`, not the Anthropic SDK), `OllamaLLM`, `CustomLLM` (OpenAI-compatible). Factory: `create_llm_client(config.llm)`.
- `prompts/planning.txt` — system prompt for plan generation. The LLM must return strict JSON; `repair_llm_response()` in `client.py` handles common formatting errors (markdown fences, trailing commas, missing `recommended` field).

### Adding a new browser platform

1. Create `engine/modules/browser/<platform>.py` implementing all abstract methods from `BrowserAutomation`
2. Add a loader function and entry in `APP_REGISTRY` / `_LOADERS` in `registry.py`
3. Un-comment the entry in `APP_REGISTRY`

### Adding a new LLM provider

1. Subclass `LLMClient` in `engine/modules/llm/client.py`
2. Add its config model in `engine/config.py`
3. Add a branch in `create_llm_client()`

### Key env vars

| Variable | Required for |
|---|---|
| `ANTHROPIC_API_KEY` | LLM planning (primary) |
| `ELEVENLABS_API_KEY` | TTS (Phase 4+) |
| `OPENAI_API_KEY` | OpenAI LLM fallback |
| `AVATAR_CONFIG` | Override config file path |

### Logging

Structured JSON logs via `structlog`. Use `get_logger("module_name")` from `engine/logging_config.py`. Log output goes to stdout and `./logs/agent.log`. All state transitions are logged at `INFO` with `state_transition` as the event key.

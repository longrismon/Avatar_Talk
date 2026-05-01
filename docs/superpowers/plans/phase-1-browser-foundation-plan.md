# Phase 1: Browser Automation Foundation — Implementation Plan

**Source:** ADR-002 (Avatar Talk project, C:/Users/lenguyen/Work/Avatar_Talk/)
**Goal:** Reliable, recoverable browser automation for Teams. CLI can open Teams, find a contact, read chat history, and start a call.
**Exit criteria:** `python avatar.py plan "Call Alex on Teams to schedule a meeting"` executes the full pre-call sequence reliably.

---

## Project Context

This is a new project being scaffolded from scratch. The working directory is `C:/Users/lenguyen/Work/Avatar_Talk/`. The only existing files are two ADR documents.

The system: An AI agent that automates communication platforms (Teams, Slack), conducts live voice calls via STT→LLM→TTS pipeline, with human-in-the-loop approval before each AI response.

Phase 1 covers only the browser automation foundation. Audio, LLM, and UI come in later phases.

**Stack:**
- Python 3.11+ with asyncio
- Playwright (Python async) for browser automation
- Pydantic v2 + YAML for config
- structlog for structured JSON logging

---

## Task 1: Scaffold Monorepo Directory Structure

Create the complete directory skeleton for the entire project (empty `__init__.py` files, placeholder dirs). No code yet — just the bones.

**Directory structure to create** (under `C:/Users/lenguyen/Work/Avatar_Talk/`):

```
engine/
  __init__.py
  main.py                     (entry point — minimal stub)
  orchestrator/
    __init__.py
    state_machine.py          (stub)
    events.py                 (stub)
  modules/
    __init__.py
    audio/
      __init__.py
      stt_interface.py        (stub)
      tts_interface.py        (stub)
      vad.py                  (stub)
      virtual_devices.py      (stub)
    browser/
      __init__.py
      interface.py            (stub)
      registry.py             (stub)
      teams.py                (stub)
    llm/
      __init__.py
      interface.py            (stub)
      client.py               (stub)
      prompts/
        conversation.txt      (empty)
        planning.txt          (empty)
        summary.txt           (empty)
    lipsync/
      __init__.py
      interface.py            (stub)
      virtual_camera.py       (stub)
    hitl/
      __init__.py
      controller.py           (stub)
  server/
    __init__.py
    websocket.py              (stub)
    http.py                   (stub)
    auth.py                   (stub)
  profiles/
    default.json              (template profile)
  tests/
    __init__.py
    test_state_machine.py     (stub)
    test_browser_teams.py     (stub)
    test_audio_pipeline.py    (stub)
    test_llm_responses.py     (stub)
engine/pyproject.toml
engine/config.yaml

docs/
  architecture.md             (stub)
  setup-guide.md              (stub)
  api-reference.md            (stub)
scripts/
  setup-linux.sh              (stub)
  setup-windows.ps1           (stub)
  download-models.py          (stub)
```

**Deliverables:**
- All directories and files created
- `pyproject.toml` with proper dependencies listed (NOT installed — just declared)
- `config.yaml` with full config schema (from spec Section 8)
- `profiles/default.json` with principal profile template (from spec Appendix A)

**pyproject.toml must declare these dependencies:**
```toml
[project]
name = "avatar-engine"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "structlog>=23.0",
    "playwright>=1.40",
    "websockets>=12.0",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "httpx>=0.27",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
stt = ["faster-whisper>=1.0", "sounddevice>=0.4"]
tts = ["sounddevice>=0.4"]
ml = ["torch>=2.0", "silero-vad"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "pytest-mock>=3.12"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"
```

---

## Task 2: Configuration System

Implement the Pydantic v2 config system that loads `config.yaml` and validates all settings.

**Files to implement:**
- `engine/config.py` — the config module

**Requirements:**
- `Settings` root model with nested sub-models: `ServerConfig`, `ReviewConfig`, `AudioConfig`, `STTConfig`, `TTSConfig`, `VADConfig`, `LLMConfig`, `LipSyncConfig`, `BrowserConfig`, `VirtualDevicesConfig`, `NotificationsConfig`, `LoggingConfig`
- Load from `config.yaml` using PyYAML
- Support `${ENV_VAR}` interpolation in string values (replace `${FOO}` with `os.environ["FOO"]`)
- `load_config(path: str) -> Settings` function
- Validation: port range 1024-65535, timeout > 0, model names non-empty
- `config.yaml` must match the full schema from spec Section 8

**Config model structure (key fields):**
```python
class ServerConfig(BaseModel):
    port: int  # 1024-65535
    host: str
    auth_token: str
    mdns_enabled: bool
    mdns_name: str

class ReviewConfig(BaseModel):
    timeout_seconds: float  # 2.0-15.0
    auto_select_recommended: bool
    keyboard_shortcuts_enabled: bool

class STTConfig(BaseModel):
    primary: Literal["faster-whisper", "google-stt"]
    fallback: Literal["faster-whisper", "google-stt"]
    whisper: WhisperConfig
    google: GoogleSTTConfig

# ... etc for all sub-configs
```

**Test:** `test_config.py` with tests for:
- Loads valid config without errors
- `${ENV_VAR}` interpolation works
- Invalid port raises ValidationError
- Missing required field raises ValidationError

---

## Task 3: Browser Automation Interface + Registry

Implement the abstract `BrowserAutomation` interface and `AppRegistry`.

**Files to implement:**
- `engine/modules/browser/interface.py` — abstract base class
- `engine/modules/browser/registry.py` — app registry

**Requirements for `interface.py`:**
```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod

class ActionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    NEEDS_INTERVENTION = "needs_intervention"

@dataclass
class ActionResult:
    status: ActionStatus
    data: Optional[dict] = None
    error: Optional[str] = None
    screenshot_path: Optional[str] = None

class BrowserAutomation(ABC):
    @abstractmethod
    async def open_app(self, app_name: str) -> ActionResult: ...
    @abstractmethod
    async def search_contact(self, name: str) -> ActionResult: ...
    @abstractmethod
    async def read_chat_history(self, contact: str, limit: int = 50) -> ActionResult: ...
    @abstractmethod
    async def start_call(self, contact: str, video: bool = True) -> ActionResult: ...
    @abstractmethod
    async def grant_permissions(self, mic: bool = True, camera: bool = True) -> ActionResult: ...
    @abstractmethod
    async def end_call(self) -> ActionResult: ...
    @abstractmethod
    async def send_message(self, contact: str, text: str) -> ActionResult: ...
    @abstractmethod
    async def capture_tab_audio(self) -> ActionResult: ...
    @abstractmethod
    async def take_screenshot(self) -> ActionResult: ...
```

**Requirements for `registry.py`:**
- `APP_REGISTRY: dict[str, type[BrowserAutomation]]` — maps app name to class
- `get_automation(app_name: str) -> type[BrowserAutomation]` — raises `ValueError` for unknown apps
- Register only "teams" for now (Slack/Discord as commented stubs)

**Test:** `test_browser_interface.py`:
- `get_automation("teams")` returns TeamsAutomation class
- `get_automation("unknown")` raises ValueError
- `get_automation("TEAMS")` (uppercase) works (case-insensitive)

---

## Task 4: Teams Browser Automation

Implement `TeamsAutomation` — the full Teams browser automation class using Playwright with the three-tier selector strategy from ADR-002.

**File to implement:** `engine/modules/browser/teams.py`

**Three-tier selector strategy (REQUIRED for every element interaction):**
```python
async def _click_element(self, page, description: str, selectors: list[dict]) -> None:
    """
    selectors is an ordered list of attempts:
    [
        {"type": "aria", "label": "Call"},
        {"type": "role", "role": "button", "name": "Call"},
        {"type": "visual", "description": "the green call button"},
    ]
    Tries each in order. Falls back to visual LLM lookup on last resort.
    """
```

**SELECTOR CONSTANTS** (as a class-level dict, not inline strings):
```python
SELECTORS = {
    "search_box": [
        {"type": "aria", "label": "Search"},
        {"type": "tid", "tid": "search-input"},
        {"type": "placeholder", "text": "Search"},
    ],
    "contact_result": [
        {"type": "tid", "tid": "search-result-person"},
        {"type": "role", "role": "option"},
    ],
    "call_button": [
        {"type": "aria", "label": "Call"},
        {"type": "tid", "tid": "call-button"},
        {"type": "role", "role": "button", "name": re.compile("call", re.I)},
    ],
    "hangup_button": [
        {"type": "aria", "label": "Hang up"},
        {"type": "tid", "tid": "hangup-button"},
    ],
    "chat_messages": [
        {"type": "tid", "tid": "chat-pane-message"},
        {"type": "role", "role": "listitem"},
    ],
    "permission_allow": [
        {"type": "text", "text": "Allow"},
    ],
}
```

**Methods to implement** (all async):
- `open_app(app_name: str) -> ActionResult` — navigate to teams.microsoft.com, check login state
- `search_contact(name: str) -> ActionResult` — find contact by name
- `read_chat_history(contact: str, limit: int = 50) -> ActionResult` — scrape messages, return `{"messages": [{"sender": str, "text": str, "timestamp": str}]}`
- `start_call(contact: str, video: bool = True) -> ActionResult`
- `grant_permissions(mic: bool, camera: bool) -> ActionResult`
- `end_call() -> ActionResult`
- `send_message(contact: str, text: str) -> ActionResult`
- `take_screenshot() -> ActionResult` — saves to `/tmp/avatar_agent_<timestamp>.png`, returns path in `data`
- `_screenshot() -> str` — internal helper

**Pre-flight auth check** (implement as `auth_check() -> ActionResult`):
- Navigate to `https://teams.microsoft.com`
- If home page loads (logged in): return SUCCESS
- If sign-in page appears: return NEEDS_INTERVENTION with error message "Teams requires login. Please sign in in the browser window and then retry."

**Test** (`test_browser_teams.py` using pytest-mock/AsyncMock):
- `open_app` returns NEEDS_INTERVENTION when sign-in page is detected
- `search_contact` returns FAILED when contact not found
- `read_chat_history` parses messages correctly from mock DOM
- `auth_check` detects logged-out state

---

## Task 5: Browser Pool + Playwright Lifecycle

Implement the `BrowserPool` that manages the Playwright browser instance lifecycle.

**File:** `engine/modules/browser/pool.py`

**Requirements:**
- `BrowserPool` manages a single persistent Chromium browser instance
- Uses `user_data_dir` from config for session persistence
- `async def get_automation(app_name: str) -> BrowserAutomation` — returns automation instance, lazily creating browser if needed
- `async def setup()` — launch Playwright + Chromium with the right flags:
  ```python
  browser = await playwright.chromium.launch_persistent_context(
      user_data_dir=config.browser.user_data_dir,
      headless=config.browser.headless,
      args=[
          "--use-fake-device-for-media-stream",  # for virtual audio/cam
          "--autoplay-policy=no-user-gesture-required",
          "--disable-blink-features=AutomationControlled",
      ]
  )
  ```
- `async def teardown()` — close browser cleanly
- `async def run_preflight_check() -> ActionResult` — runs auth check for the default app

**Note:** On Windows, Playwright uses Chromium (not Chrome). The `user_data_dir` path must be Windows-compatible.

---

## Task 6: Logging Infrastructure

Implement structured logging for the entire engine using `structlog`.

**File:** `engine/logging_config.py`

**Requirements:**
- Configure structlog with JSON output (one JSON object per line)
- Include these fields on every log entry: `timestamp_ms`, `module`, `level`
- `setup_logging(level: str, log_file: str | None)` function
- `get_logger(module: str)` returns a bound logger with `module` pre-set
- Log format matches the `LogEvent` schema from ADR-002:
  ```python
  # Every log event should support these fields:
  # timestamp_ms, call_id, turn_number, module, event_type, duration_ms, payload, level
  ```
- File rotation: max 50MB, 5 backups (use `logging.handlers.RotatingFileHandler`)

---

## Task 7: CLI Entry Point + Planning LLM Call

Implement the CLI entry point and the LLM planning module.

**Files:**
- `engine/main.py` — CLI with `plan` subcommand
- `engine/modules/llm/client.py` — `AnthropicLLM.generate_plan()` only (other methods stubbed)

**CLI requirements** (`python avatar.py plan "Call Alex on Teams"`):
```
Usage: avatar.py plan <instruction>
       avatar.py health
       avatar.py version

Options:
  --config PATH   Config file path [default: config.yaml]
  --dry-run       Show plan without executing
```

- `plan` command: loads config, calls `AnthropicLLM.generate_plan()`, prints the plan as formatted JSON, then asks "Execute this plan? [y/N]"
- `health` command: prints engine component status
- `version` command: prints version

**`generate_plan()` implementation:**
- Uses the planning system prompt from `engine/modules/llm/prompts/planning.txt`
- Implement `repair_llm_response()` JSON repair function (from ADR-002 Decision 6)
- Handles `${ANTHROPIC_API_KEY}` env var
- Returns `ActionPlan` dataclass with `steps`, `mission_summary`, `estimated_duration`, `conversation_goal`, `success_criteria`

**Planning system prompt** (`engine/modules/llm/prompts/planning.txt`):
Use the exact planning system prompt from Section 13 of the original spec.

**Test** (`test_llm_planning.py`):
- `repair_llm_response()` handles missing `recommended` field
- `repair_llm_response()` handles trailing commas in JSON
- `repair_llm_response()` strips markdown fences
- `generate_plan()` returns correct ActionPlan structure (mock the HTTP call)

---

## Task 8: State Machine Core (Phase 1 States Only)

Implement the `Orchestrator` state machine, but only the Phase 1 states: IDLE → PLANNING → BROWSER_ACTION → CALL_ENDED/ERROR loop.

**File:** `engine/orchestrator/state_machine.py`

**States to implement (Phase 1):**
- `IDLE` — waits for instruction
- `PLANNING` — calls LLM to generate action plan, presents it to user for confirmation
- `BROWSER_ACTION` — executes each step in the plan sequentially
- `CALL_ENDED` — (stub — just logs and returns to IDLE)
- `ERROR` — handles failures, offers retry/abort

**State machine requirements:**
- `AgentState` enum with ALL 11 states (even if most are stubs)
- `StateContext` dataclass
- `Orchestrator` class with `async def run()` main loop
- `async def transition(new_state, **kwargs)` logs every transition to structlog
- `async def emit(event: dict)` — for now just logs; later will broadcast to WebSocket
- Event waiting via `asyncio.Event` + a simple in-process event queue
- `handle_ui_event(event: dict)` for processing UI commands

**BROWSER_ACTION handler must:**
- Execute steps from `ctx.action_plan` sequentially
- Handle `SUCCESS`, `FAILED`, `NEEDS_INTERVENTION` results
- Print step progress to stdout (for Phase 1 CLI mode)
- Retry once on `NEEDS_INTERVENTION` after printing the intervention message

**Test** (`test_state_machine.py`):
- IDLE → PLANNING transition works
- PLANNING → BROWSER_ACTION on plan confirmed
- PLANNING → IDLE on plan rejected
- BROWSER_ACTION → ERROR on step failure
- ERROR → IDLE on abort

---

## Task 9: Integration Test + README

Wire everything together and verify the Phase 1 exit criteria.

**Files:**
- `engine/tests/test_integration_phase1.py`
- `README.md` (project root)

**Integration test:**
Using `pytest` + `AsyncMock` to mock the Playwright browser, test the full flow:
1. Create Orchestrator with mocked BrowserPool
2. Send `user_instruction` event: "Call Alex on Teams"
3. Verify state transitions: IDLE → PLANNING → BROWSER_ACTION → (mock call connected)
4. Verify all browser actions are called in sequence
5. Verify structured log events are emitted

**README.md must cover:**
- Project overview (1 paragraph)
- Prerequisites (Python 3.11+, Playwright, API keys)
- Quick start: `pip install -e ".[dev]" && python avatar.py plan "Call Alex on Teams"`
- Phase 1 limitations (no audio, no LLM during call, no video)
- Ethics & ToS notice (from ADR-002 Section "Terms of Service and Ethics")
- Link to ADR docs

---

## Summary

| # | Task | Files | Dependencies |
|---|------|-------|-------------|
| 1 | Scaffold monorepo | All dirs + stubs | None |
| 2 | Config system | `engine/config.py`, `config.yaml` | Task 1 |
| 3 | Browser interface + registry | `interface.py`, `registry.py` | Task 1 |
| 4 | Teams automation | `teams.py` | Task 3 |
| 5 | Browser pool | `pool.py` | Tasks 3, 4 |
| 6 | Logging | `logging_config.py` | Task 1 |
| 7 | CLI + planning LLM | `main.py`, `client.py` | Tasks 2, 6 |
| 8 | State machine | `state_machine.py` | Tasks 5, 6, 7 |
| 9 | Integration + README | `test_integration_phase1.py`, `README.md` | All |

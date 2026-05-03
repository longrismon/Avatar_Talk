# Avatar Agent

An AI agent that conducts live voice calls on your behalf. It listens to the other party via speech-to-text, generates response candidates using an LLM, lets you review and approve them within a configurable window, speaks the chosen response via TTS, and displays a lip-synced avatar of your face — all supervised from a React web UI or your mobile phone.

**All 6 phases implemented.**

---

## How It Works

```
Other party speaks
        ↓
   VAD (Silero) detects utterance boundary
        ↓
   Whisper STT transcribes speech
        ↓
   LLM (Claude) generates 4 response options  ←── speculative pre-gen starts
        ↓                                           immediately after you speak
   Review UI — 4 options + countdown timer
        ↓   (keyboard 1–4 or auto-select on timeout)
   ElevenLabs TTS synthesizes chosen response
        ↓
   Virtual mic injects audio into call  ──────────┐
   Wav2Lip / SadTalker renders lip-sync video ────┘  (parallel)
        ↓
   Mobile push notification sent (Firebase FCM)
        ↓
   Loop back to listening
```

---

## Prerequisites

| Requirement | Used for |
|---|---|
| Python 3.11+ | Runtime |
| [Anthropic API key](https://console.anthropic.com/) | LLM response generation & planning |
| [ElevenLabs API key](https://elevenlabs.io/) | TTS voice synthesis |
| Microsoft Teams account (web) | Call automation |
| [Playwright](https://playwright.dev/) Chromium | Browser automation |
| Node.js 18+ | React review UI |

**Optional:**

| Requirement | Used for |
|---|---|
| NVIDIA GPU + CUDA | Wav2Lip lip-sync, local Whisper STT |
| Google Cloud Speech credentials | Fallback STT |
| Firebase service account JSON | Mobile push notifications |
| BlackHole (macOS) / VB-Cable (Windows) | Virtual audio device |
| v4l2loopback (Linux) | Virtual camera device |

---

## Quick Start

```bash
# 1. Install Python dependencies
cd engine
pip install -e ".[dev]"
playwright install chromium

# 2. Set API keys
export ANTHROPIC_API_KEY="your-key"
export ELEVENLABS_API_KEY="your-key"

# 3. Edit config and profile
#    engine/config.yaml        — server, LLM, audio, lipsync, notifications settings
#    engine/profiles/default.json — your name, role, tone, topics to avoid

# 4. Install virtual audio/camera devices (Linux)
chmod +x engine/setup/install_virtual_devices.sh
./engine/setup/install_virtual_devices.sh
# macOS/Windows: the script prints manual instructions for BlackHole / VB-Cable

# 5. Build the React UI
cd frontend && npm install && npm run build && cd ..

# 6. Run the agent server
python -c "
import uvicorn
from engine.config import load_config
from engine.modules.llm.client import create_llm_client
from engine.orchestrator.state_machine import Orchestrator
from engine.server.http import create_app

config = load_config('engine/config.yaml')
orch = Orchestrator(
    llm=create_llm_client(config.llm),
    review_config=config.review,
)
app = create_app(orch, config)
uvicorn.run(app, host='0.0.0.0', port=9600)
"
# Open http://localhost:9600 — React UI loads
```

**CLI commands:**

```bash
# Dry-run: generate a plan without opening a browser
python avatar.py plan "Call Alex on Teams to schedule a meeting" --dry-run

# Execute a full plan (opens browser, runs the call)
python avatar.py plan "Call Alex on Teams to schedule a meeting"

# Synthesize text and inject into virtual mic (TTS demo)
python avatar.py speak "Hello, thanks for joining today."

# Check engine component status and API key presence
python avatar.py health

# Smoke-test browser selectors for a platform
python avatar.py selector-health --app teams

# Show version
python avatar.py version
```

---

## Configuration

All settings live in `engine/config.yaml`. Environment variable substitution is supported with `${VAR}` syntax.

**Key sections:**

```yaml
server:
  port: 9600
  auth_token: "change-me-in-production"

llm:
  primary: anthropic          # anthropic | openai | ollama | custom
  fallback: ollama
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-sonnet-4-20250514"

audio:
  stt:
    primary: faster-whisper   # faster-whisper | google-stt
  tts:
    primary: elevenlabs
    elevenlabs:
      api_key: "${ELEVENLABS_API_KEY}"
      voice_id: "your-voice-id"

review:
  timeout_seconds: 5.0        # auto-selects recommended after this many seconds
  auto_select_recommended: true

lipsync:
  enabled: false              # set true to enable Wav2Lip / SadTalker
  engine: wav2lip
  reference_image: "./face/reference.png"

notifications:
  push_enabled: false         # set true + fill firebase credentials to enable FCM
  firebase:
    credentials_path: "./credentials/firebase-admin.json"
    device_token: ""          # FCM registration token of your mobile device
```

**Principal profile** (`engine/profiles/default.json`):

```json
{
  "name": "Your Name",
  "role": "Senior Engineer",
  "tone": "professional and friendly",
  "avoid_topics": ["salary", "personal matters"],
  "speaking_style": "direct and concise"
}
```

This profile is injected into every LLM prompt so responses match your voice and constraints.

---

## Review UI

The React SPA (served from `http://localhost:9600`) displays:

- **4 response options** generated by the LLM with different tones: professional / empathetic / direct / light
- **Countdown timer** — auto-selects the recommended option when it expires
- **Keyboard shortcuts** — press `1`–`4` to select, `T` or `Esc` to take over manually
- **Status bar** — live call timer and current agent state (listening / generating / speaking)

The review timeout adapts to conversation phase: 1.5× longer early in a call, standard mid-call, 0.75× shorter near the end.

---

## Performance Monitoring

The orchestrator tracks per-turn latencies via `PerformanceTracker`. Access it at runtime:

```python
orch.perf.summary()
# {
#   "end_to_end": {"p50": 3200, "p95": 5800, "p99": 7100, "count": 42},
#   "llm":        {"p50": 1800, "p95": 3200, "p99": 4100, "count": 42},
#   "tts":        {"p50": 600,  "p95": 1100, "p99": 1400, "count": 42},
#   "total_turn": {"p50": 5100, "p95": 8900, "p99": 10200, "count": 42},
#   "turns_recorded": 42
# }
```

Stages tracked: `utterance_end → llm_start → llm_complete → tts_first_chunk → speaking_complete`.

---

## Speculative Pre-Generation

Immediately after the agent finishes speaking, a background LLM call starts using the current transcript context. If it completes before the other party finishes their next utterance, the GENERATING state consumes it instantly — giving near-zero LLM wait time on predictable turns.

---

## Mobile Notifications (Firebase FCM)

Set `notifications.push_enabled: true` and provide a Firebase service-account JSON plus your device's FCM registration token. The agent pushes:

| Event | When |
|---|---|
| `call_connected` | The call connects and the audio pipeline starts |
| `review_started` | Response options are ready for your review |
| `intervention_needed` | A browser automation step needs manual attention |
| `call_ended` | The call ends; notification body contains the LLM-generated call summary |

---

## Project Structure

```
avatar-agent/
├── avatar.py                        # CLI entry point
├── engine/
│   ├── config.py                    # Pydantic v2 settings models
│   ├── config.yaml                  # Runtime configuration (edit this)
│   ├── logging_config.py            # structlog JSON logging
│   ├── profiles/
│   │   └── default.json             # Principal persona profile (edit this)
│   ├── orchestrator/
│   │   └── state_machine.py         # Central FSM + all state handlers
│   ├── modules/
│   │   ├── browser/
│   │   │   ├── interface.py         # BrowserAutomation ABC
│   │   │   ├── pool.py              # Persistent Playwright context
│   │   │   ├── teams.py             # Teams automation (three-tier selectors)
│   │   │   ├── registry.py          # App name → class registry
│   │   │   └── health_check.py      # Selector smoke-test
│   │   ├── llm/
│   │   │   ├── interface.py         # LLMClient ABC
│   │   │   ├── client.py            # AnthropicLLM, OllamaLLM, CustomLLM
│   │   │   └── prompts/             # System prompts (planning, conversation, summary)
│   │   ├── audio/
│   │   │   ├── stt.py               # Faster-Whisper + Google STT
│   │   │   ├── tts.py               # ElevenLabs + Coqui XTTS
│   │   │   ├── vad.py               # Silero VAD
│   │   │   ├── cache.py             # Pre-baked filler audio cache
│   │   │   └── virtual_devices.py   # Virtual mic injection
│   │   ├── lipsync/
│   │   │   ├── interface.py         # LipSyncClient ABC
│   │   │   ├── wav2lip.py           # Wav2Lip inference (real-time)
│   │   │   ├── sadtalker.py         # SadTalker inference (offline)
│   │   │   └── virtual_camera.py    # v4l2loopback video injection
│   │   ├── notifications/
│   │   │   ├── interface.py         # NotificationClient ABC
│   │   │   └── firebase.py          # Firebase FCM sender
│   │   └── performance.py           # Per-turn latency tracker (p50/p95/p99)
│   ├── server/
│   │   ├── http.py                  # FastAPI app factory
│   │   ├── websocket.py             # WebSocket ConnectionManager
│   │   └── auth.py                  # Bearer token verification
│   ├── setup/
│   │   └── install_virtual_devices.sh  # One-command Linux/macOS/Windows device setup
│   └── tests/                       # 224 tests across all modules
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Top-level state machine (idle/reviewing/speaking)
│   │   ├── components/
│   │   │   ├── ReviewPanel.jsx      # 4 options + countdown + keyboard shortcuts
│   │   │   └── StatusBar.jsx        # Call timer + agent state indicator
│   │   └── hooks/
│   │       └── useWebSocket.js      # WS connection + auth + 20s ping keepalive
│   └── dist/                        # Built React app (served by FastAPI)
├── ADR-001-AI-Avatar-Agent-Architecture.md
└── ADR-002-Production-Build-Plan.md
```

---

## Phase Summary

| Phase | What was added |
|---|---|
| 1 — Browser Foundation | Playwright Teams automation, LLM planning, state machine (IDLE→PLANNING→BROWSER_ACTION) |
| 2 — Audio Pipeline | Silero VAD, faster-whisper STT, ElevenLabs/Coqui TTS, virtual mic injection |
| 3 — LLM + HITL Review | Two-temperature response generation, React review UI, FastAPI/WebSocket server |
| 4 — Lip-Sync Video | Wav2Lip + SadTalker, v4l2loopback virtual camera, parallel audio+video during SPEAKING |
| 5 — Mobile Notifications | Firebase FCM push, final call summary, four notification events |
| 6 — Polish & Hardening | `call_id` logging, p50/p95/p99 perf tracking, adaptive review timeout, speculative pre-gen, selector health monitor, virtual device installer |

---

## Running Tests

```bash
cd engine
pytest tests/ -v            # 224 tests across all modules
pytest tests/ -v -k llm     # filter by keyword
```

---

## WebSocket Protocol

The React UI connects to `ws://localhost:9600/ws?token=<auth_token>`.

**Client → Server:**

| Message | Effect |
|---|---|
| `{"type": "selection", "option_id": 2}` | Select response option 2 |
| `{"type": "takeover"}` | Switch to MANUAL_OVERRIDE state |
| `{"type": "resume_ai"}` | Resume AI from MANUAL_OVERRIDE |
| `{"type": "ping"}` | Server replies `{"type": "pong"}` |

**Server → Client:**

| Event | When |
|---|---|
| `state_changed` | Any FSM state transition (includes full context snapshot) |
| `options` | 4 LLM response options ready |
| `review_started` | HUMAN_REVIEW entered (includes `timeout_seconds`) |
| `response_selected` | A response was chosen (human or auto-select) |
| `speaking` / `speaking_complete` | TTS playback start/end |
| `utterance_complete` | Other party finished an utterance |
| `call_ended` | Call ended with `final_summary` |
| `error` | Agent error with `source_state` |

---

## Selector Health Check

Smoke-test browser selectors to catch UI drift before a live call:

```bash
python avatar.py selector-health --app teams
#   ✓ chat_nav        42ms
#   ✓ teams_nav       38ms
#   ✓ search_box      91ms
#   ✗ message_input  5003ms  (Timeout waiting for element)
#
#   5 passed, 1 failed
```

Re-run after Teams/Slack updates. All selectors use a three-tier fallback (`aria-label` → `data-tid` → role/placeholder) to maximize resilience.

---

## Extending the System

**Add a new LLM provider:**
1. Subclass `LLMClient` in `engine/modules/llm/client.py`; implement `_chat()`, `generate_responses()`, `summarize_call()`, `generate_plan()`
2. Add a config model in `engine/config.py` and extend `LLMConfig`
3. Add a branch in `create_llm_client()`

**Add a new browser platform:**
1. Create `engine/modules/browser/<platform>.py` implementing all abstract methods from `BrowserAutomation`
2. Register it in `engine/modules/browser/registry.py`
3. Add its selectors to `engine/modules/browser/health_check.py`

**Add a new notification provider:**
1. Subclass `NotificationClient` in `engine/modules/notifications/`
2. Update `create_notifier()` in `engine/modules/notifications/__init__.py`

---

## Ethics & Terms of Service

By using this software you acknowledge:

1. **Terms of Service** — Automating Microsoft Teams, Slack, or Discord may violate their Terms of Service. Use only with accounts you own and at your own risk.
2. **Voice synthesis** — TTS uses your own voice profile. Never use this system to impersonate anyone other than yourself.
3. **Disclosure** — The other party on the call does not know they are interacting with an AI-assisted system. Consider whether disclosure is required or appropriate in your context and jurisdiction.
4. **Recording consent** — Transcribing calls may require all-party consent. Check your local laws before use.

---

## Architecture Docs

- [ADR-001](ADR-001-AI-Avatar-Agent-Architecture.md) — Technology evaluation, design decisions, and risk analysis
- [ADR-002](ADR-002-Production-Build-Plan.md) — Six-phase production build plan with exit criteria and timelines

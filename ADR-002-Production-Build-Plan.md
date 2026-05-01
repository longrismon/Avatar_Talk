# ADR-002: AI Avatar Agent — Production Architecture & Build Plan

**Status:** Accepted
**Date:** 2026-04-15
**Supersedes:** ADR-001 (evaluation only)
**Deciders:** Engineering Lead, Product Owner
**Horizon:** 13-week build, 6 phases

---

## Context

ADR-001 evaluated the feasibility and risks of the AI Avatar Agent specification. This document records the production architecture decisions and serves as the engineering blueprint for the full build.

The system must:
- Automate browser-based communication platforms (initially Teams)
- Conduct live voice calls on behalf of the principal with a 5-second human review window
- Inject AI-synthesized voice (and optionally face video) into the call
- Allow instant principal takeover at any moment
- Summarize calls and track mission completion

The minimum viable product is **Phase 3** (audio-only conversation loop with human approval, no LipSync). Everything beyond that is enhancement.

---

## Decision 1: Deployment Topology — Single Machine vs. Split

### Options

**Option A: Single machine (RTX 3090 or better)**
All six modules run on one host. Python `asyncio` is the concurrency model. Inter-module communication via in-process async queues (`asyncio.Queue`).

| Dimension | Assessment |
|---|---|
| Complexity | Low — no network between modules |
| Latency | Lowest — no serialization overhead |
| Cost | One-time GPU hardware cost |
| Operational risk | Single point of failure |
| Hardware requirement | RTX 3090 (24GB VRAM) minimum for all models loaded |

**Option B: Split — local machine (browser + UI) + GPU server (ML models)**
Browser automation and control UI on the user's laptop. STT, TTS, LipSync, and LLM proxy on a remote GPU server. Communication over WebSocket.

| Dimension | Assessment |
|---|---|
| Complexity | High — network protocol between modules, latency budget tighter |
| Latency | +50–150ms round-trip per pipeline stage |
| Cost | Ongoing cloud GPU cost (~$1–3/hr for A100/H100) |
| Operational risk | Network failure during call is catastrophic |
| Hardware requirement | Any laptop for local; cloud GPU for inference |

### Decision: **Option A for v1, Option B as v2 scale path**

A single RTX 3090 can handle the full stack:
- faster-whisper large-v3: ~4GB VRAM
- ElevenLabs: API (no local GPU)
- Wav2Lip (v2 LipSync): ~2GB VRAM
- All models + browser: well within 24GB

The latency saved by staying in-process is meaningful given our tight budget. Option B becomes relevant only when the principal wants to run this on a laptop without a GPU, or when multi-user/concurrent call support is needed.

**Inter-module communication: `asyncio.Queue` (in-process).** No Redis, no network broker for v1. The event bus is a set of typed async queues between modules. This is simpler to debug, has zero serialization overhead, and requires no external service. If we later move to Option B, we swap the queues for WebSocket channels with the same interface.

---

## Decision 2: Event Bus & Orchestrator Pattern

### Options

**Option A: Centralized state machine (explicit FSM)**
A single `Orchestrator` class owns the state enum and all transition logic. Each module calls `orchestrator.emit(event)`. Transitions are a lookup table.

**Option B: Actor model / message passing**
Each module is an independent async actor with its own inbox queue. The orchestrator is just another actor that subscribes to module outputs. No shared state.

**Option C: Workflow engine (Temporal, Prefect)**
Use an external workflow engine to define the state machine as a durable workflow with built-in retry, timeout, and replay.

### Decision: **Option A (centralized FSM) for v1**

Option C adds an external dependency that is overkill for a single-user application. Option B makes debugging harder — when something goes wrong mid-call, you need to understand the full state at that moment, which is easier with a single authoritative state object.

The `Orchestrator` class:
```python
class Orchestrator:
    state: AgentState          # enum with all 11 states
    mission: Mission           # from PLANNING
    context: ConversationContext  # accumulated transcript
    queues: ModuleQueues       # typed async queues to/from each module

    async def emit(self, event: AgentEvent) -> None:
        transition = TRANSITION_TABLE[(self.state, event.type)]
        await transition.handler(self, event)
        self.state = transition.next_state
        await self._log_transition(event, transition)
```

All transitions are logged with timestamps to a `call_{id}.jsonl` file for post-call debugging and replay.

---

## Decision 3: Browser Automation Strategy

### The Core Problem

Teams web UI uses dynamic CSS classes (e.g., `css-1a2b3c`) that change with every deploy. The spec references these via CSS selectors, which will break within weeks.

### Decision: **Semantic locator hierarchy with screenshot fallback**

Use a three-tier selector strategy for every element:

1. **Semantic (preferred):** `aria-label`, `data-tid`, `role` attributes — these are stable across UI updates.
2. **Text-based fallback:** `page.get_by_text("Call")`, `page.get_by_role("button", name="Video call")` — less fragile than CSS.
3. **Visual fallback (last resort):** Take a screenshot, describe the target element to the LLM, get back coordinates to click. Expensive (~1s) but recovers from total selector failure.

```python
async def click_call_button(page: Page) -> None:
    try:
        # Tier 1: semantic
        await page.get_by_label("Call").click(timeout=3000)
    except PlaywrightTimeoutError:
        try:
            # Tier 2: text
            await page.get_by_role("button", name=re.compile("call", re.I)).click(timeout=3000)
        except PlaywrightTimeoutError:
            # Tier 3: visual LLM fallback
            coords = await visual_find_element(page, "the green call button")
            await page.mouse.click(*coords)
```

### Authentication management

Use a persistent Chromium profile (`user_data_dir`). Run a **pre-flight check** at every startup:
1. Navigate to Teams, check if the home page loads (user is logged in).
2. If the sign-in page appears, pause and notify the principal to log in manually via the real browser window (don't automate SSO — too fragile and ToS-risky).
3. Block the call from starting until auth is confirmed.

This means the first-run experience requires the user to log in once manually. After that, the session persists.

### Virtual device setup (required before first call)

Document this as a setup prerequisite, not runtime logic:

**Linux (recommended for production):**
```bash
# Virtual mic
sudo modprobe snd-aloop index=10,11 enable=1,1 pcm_substreams=1,1
# Virtual camera (for LipSync v2)
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="AvatarCam"
```

**Windows:** VB-Cable (virtual audio) + OBS Virtual Camera. The installer must handle this.

**macOS:** BlackHole 2ch (virtual audio). LipSync camera injection is harder on macOS — document as unsupported in v1.

---

## Decision 4: STT Configuration

### Decision: **faster-whisper `large-v3` as primary, Google STT as fallback, with GPU auto-detection**

The system checks for a CUDA-capable GPU at startup:

```python
import torch
HAS_GPU = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9

STT_PRIMARY = "faster-whisper" if HAS_GPU else "google-stt"
```

If no qualifying GPU is found, the system logs a warning and switches to Google STT silently. The `config.yaml` `stt.primary` field is an override, but auto-detection is the default.

**Streaming strategy:** faster-whisper does not natively stream — it transcribes chunks. Use a 500ms sliding window with 100ms step: feed 500ms of audio to Whisper every 100ms, take the last N words as the live partial transcript. This gives a quasi-streaming experience.

For the finalized utterance (VAD silence threshold hit), run a fresh full-utterance transcription — this gives the highest accuracy for the LLM input.

**VAD:** Use `silero-vad` (PyTorch, fast, accurate) rather than energy-threshold VAD. It dramatically reduces false `utterance_complete` events from breathing, rustling, and background noise.

---

## Decision 5: TTS & Audio Injection

### Decision: **ElevenLabs streaming (primary), Coqui XTTS (fallback), pre-baked audio cache (emergency fallback)**

Three tiers, in order:

**Tier 1 — ElevenLabs streaming (latency_optimization: 3 or 4)**
- Use the WebSocket streaming API, not the REST API. First audio chunk arrives in ~250–350ms.
- Buffer audio chunks in a `asyncio.Queue` and feed them to the virtual mic device in real time.
- Track cumulative playback progress to detect `playback_complete`.

**Tier 2 — Coqui XTTS v2 (local fallback)**
- Activated if ElevenLabs API fails or is unavailable.
- Slower (800–1500ms first chunk) but fully local.
- Voice sample required at setup: minimum 10 seconds of clean speech.

**Tier 3 — Pre-baked WAV cache (emergency fallback)**
Pre-synthesize ~20 generic responses at setup time (in the principal's voice, using ElevenLabs) and cache them as WAV files:
- "Let me think about that for a second."
- "That's a good point."
- "Could you say that again?"
- "I'll have to get back to you on that."
- "Sounds good to me."
- ... (15 more)

If both live TTS engines fail, pick the most contextually appropriate cached response and play it. This ensures the call never goes completely silent due to a TTS failure.

**Audio injection to virtual mic:**

```python
async def inject_audio(audio_chunks: AsyncIterator[bytes], device: str) -> None:
    async with sounddevice.RawOutputStream(
        device=device, samplerate=24000, channels=1, dtype='int16'
    ) as stream:
        async for chunk in audio_chunks:
            await asyncio.to_thread(stream.write, chunk)
```

---

## Decision 6: LLM Architecture

### Context payload size management

For a call with N turns, the `call_transcript` grows unboundedly. The solution is a **rolling window + incremental summary**:

- Keep the last 8 turns verbatim in `call_transcript` (sufficient for immediate conversational coherence).
- Every 5 turns, run a background LLM call with a lightweight summarization prompt to update `call_summary_so_far`.
- The full context sent each turn = `principal_profile` + `mission` + `call_summary_so_far` + last 8 turns + `current_utterance`.
- Estimated token count: ~2,000–3,500 tokens per turn (well within limits).

### Two-temperature response generation

The current spec uses `temperature: 0.7` for everything. The risk: the LLM's recommended option (auto-selected after 5 seconds) is non-deterministic at 0.7, which could produce an embarrassing auto-selection.

**Decision:** Use two API calls per turn:
1. **Option generation** (`temperature: 0.7`, `max_tokens: 800`): Generate 4 diverse candidate options.
2. **Recommendation selection** (`temperature: 0.0`, `max_tokens: 50`): Given the 4 options and mission state, deterministically select which option index to recommend and why.

This separates creativity (diverse options) from safety (deterministic recommendation). The second call is cheap (~50 tokens output) and adds ~200ms, which is negligible.

### JSON repair without re-prompting

Implement a repair function that handles common LLM formatting failures before considering a re-prompt:

```python
def repair_llm_response(raw: str) -> dict:
    # Strip markdown fences
    text = re.sub(r'^```json\s*|\s*```$', '', raw.strip())
    # Fix trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Parse
    data = json.loads(text)  # raises if still broken
    # Structural repairs
    if len(data["options"]) > 4:
        data["options"] = data["options"][:4]
    if not any(o["recommended"] for o in data["options"]):
        data["options"][0]["recommended"] = True
    for o in data["options"]:
        if "recommended" not in o:
            o["recommended"] = False
    return data
```

Only re-prompt if `json.loads` still raises after repair. Reserve API-call re-prompts for truly broken outputs.

---

## Decision 7: Control UI

### Options

**Option A: React web dashboard (localhost:8765)**
Full-featured, keyboard + mouse + touch, easy to style for fast visual scanning.

**Option B: Textual terminal UI**
Fast to build, no Node.js dependency, good keyboard support.

**Option C: Electron desktop app**
Best OS integration (system tray, native notifications, always-on-top window), but heaviest to build.

### Decision: **Option A (React) for v1**

The principal needs to scan 4 options and select within 5 seconds. Visual hierarchy, color coding, and a prominent countdown timer are important for speed. A terminal UI is harder to scan quickly. Electron is premature complexity for v1.

The React app is a single-page app served by the Python backend (FastAPI + WebSocket). It communicates with the orchestrator via WebSocket:
- Server → client: `{type: "options", payload: [...]}` triggers the review panel
- Client → server: `{type: "selection", option_id: 2}` sends the selection
- Client → server: `{type: "takeover"}` triggers MANUAL_OVERRIDE

The UI must have a keyboard shortcut layer that works even when the timer is running: `1`/`2`/`3`/`4` to select, `E` to edit, `T` or `Esc` to take over. These must fire without focus on any specific element.

**UI layout (review screen):**

```
┌────────────────────────────────────────────────────┐
│  🎙 ALEX is speaking...  [call timer: 02:34]        │
│  "Sunday works. What time were you thinking?"      │
├────────────────────────────────────────────────────┤
│  ⏱  Auto-select in  [████████░░]  3s               │
├────────────────────────────────────────────────────┤
│  [1] ★  How about 2 PM? Gives you time after brunch│
│  [2]    Afternoon works — maybe 1 or 2?             │
│  [3]    Let's say after 1. I'll send a calendar...  │
│  [4]    No rush — just ping me when you're free.   │
├────────────────────────────────────────────────────┤
│  [E] Edit  [Custom]  [T] Take Over                 │
└────────────────────────────────────────────────────┘
```

---

## Decision 8: LipSync Strategy

### Decision: **Disabled in v1 ("camera off" mode). Wav2Lip in v2. MuseTalk evaluation for v3.**

Camera-off during a Teams call is socially normal and expected (roughly 60% of Teams calls have cameras off per Microsoft's own research). The audio quality of the synthesized voice is what sells the illusion, not video.

**v1:** Virtual camera shows a static profile picture (configurable). No Wav2Lip loaded at runtime, saving ~2GB VRAM and removing a major source of latency variance.

**v2 (post-Phase 4):** Wav2Lip chunked pipeline:
- Pre-load face embedding at startup (~10s).
- As TTS audio chunks stream in, generate Wav2Lip frames in parallel.
- Buffer 3–5 frames ahead before playback starts to smooth frame drops.
- If LipSync falls behind real-time, degrade to static face rather than skipping audio.

**v3 evaluation:** MuseTalk claims real-time capability with better quality than Wav2Lip. Evaluate after v2 is stable. Key metric: does it run at 25fps on an RTX 3090 with STT + TTS also running? If yes, migrate.

---

## Phased Build Plan

The 6-phase plan from the feasibility document is sound. The following refines each phase with concrete deliverables and exit criteria.

---

### Phase 1 — Browser Automation Foundation (Weeks 1–3)

**Goal:** Reliable, recoverable browser automation for Teams.

**Deliverables:**
- `BrowserModule` class with: `open_teams()`, `search_contact(name)`, `get_chat_history(contact, limit)`, `start_call(contact)`, `grant_permissions()`
- Three-tier selector strategy implemented for every element
- Pre-flight auth check at startup
- CLI: `python avatar.py plan "Call Alex on Teams to schedule a meeting"`

**Exit criteria:**
- Can start a call to a real Teams contact reliably 9/10 times without human intervention
- Auth check correctly detects logged-out state and prompts user
- Chat history scraper returns structured JSON (speaker, text, timestamp)
- All actions log to `session_{id}.jsonl`

**Key risks:**
- Teams DOM changes during the 3-week window. Mitigation: pin Teams to a specific version via browser user profile, or use the Teams desktop app via Playwright's `electron://` support (more stable DOM).
- `grant_permissions`: Chrome permission dialogs may appear as OS-level dialogs, not browser dialogs. On Linux, use xdotool. On Windows, use pyautogui as fallback.

---

### Phase 2 — Audio Pipeline (Weeks 3–5)

**Goal:** Full duplex audio: capture other party's speech, inject synthesized speech.

**Deliverables:**
- `AudioModule`: tab audio capture → STT (faster-whisper with silero-vad)
- `TTSModule`: text → ElevenLabs streaming → virtual mic injection
- Virtual device setup script (Linux/Windows)
- CLI demo: type text → hear it spoken in your voice on a live Teams call

**Exit criteria:**
- Real-time transcription latency < 2s from end-of-utterance (with GPU)
- TTS first audio chunk arrives < 500ms from text submission
- No audio echo (mic/speaker routing is clean)
- VAD correctly identifies utterance boundaries with < 0.5s false positive rate on typical office background noise

**Key implementation notes:**
- Tab audio capture: Use Playwright's `page.on("request")` to intercept WebRTC streams, or use `ffmpeg` to capture the virtual audio sink output. The latter is more reliable across platforms.
- Test with a second real Teams account in another browser window — don't rely on mock audio for pipeline validation.

---

### Phase 3 — LLM Core + Human Review UI (Weeks 5–7) ← **MVP milestone**

**Goal:** Working supervised AI conversation loop. This is the first version worth showing anyone.

**Deliverables:**
- `LLMModule`: context assembly → Claude API → JSON repair → options output
- Two-temperature generation (diversity pass + recommendation pass)
- Rolling window context management (last 8 turns + incremental summary)
- Pre-baked audio cache (20 responses, synthesized at setup)
- React control UI with WebSocket bridge
- Full keyboard shortcut layer (1/2/3/4, E, T/Esc)
- Filler deque (no repeated fillers within 5 turns)

**Exit criteria:**
- Complete turn cycle (STT → LLM → UI display → selection → TTS) under 8 seconds in the 50th percentile
- LLM produces valid options 95%+ of turns (JSON repair handles the rest)
- Principal can select an option by keyboard in under 1 second from display
- Auto-select fires correctly at timer expiry
- Take Over / Resume AI transition works instantly (< 200ms)

**Architecture note:** At Phase 3 completion, the full state machine is operational. All subsequent phases add modules *around* this core without changing it.

---

### Phase 4 — LipSync (Weeks 7–9)

**Goal:** Other party sees the principal's face "speaking."

**Deliverables:**
- `LipSyncModule`: Wav2Lip with chunked streaming pipeline
- Virtual camera device integration
- Static image fallback when LipSync can't maintain frame rate
- Graceful degradation: LipSync failure never stalls the audio pipeline

**Exit criteria:**
- Achieves 25fps Wav2Lip output with STT + TTS + LLM all running concurrently on RTX 3090
- Lip sync lag < 100ms (audio and video perceived as synchronized)
- Frame drop rate < 5% under normal operating load
- Static face fallback activates automatically when frame drop > 20%

---

### Phase 5 — Full Orchestrator + End-to-End Flow (Weeks 9–11)

**Goal:** Single command invocation handles the entire scenario from instruction to summary.

**Deliverables:**
- Complete state machine with all 11 states + recommended additions from ADR-001
- Planning module: LLM decomposes instruction into action sequence
- Post-call summarization
- Error recovery: retry logic, graceful degradation to MANUAL_OVERRIDE
- Session replay: `python avatar.py replay session_20260415_001.jsonl`

**Exit criteria:**
- Full scenario ("Call Alex on Teams to schedule a Sunday meeting") completes end-to-end with zero manual intervention 7/10 times
- Every failure mode transitions to MANUAL_OVERRIDE or ERROR with clear diagnostic message
- Post-call summary correctly identifies whether the mission goal was achieved
- Session log is complete enough to replay the call for debugging

---

### Phase 6 — Polish, Latency Optimization & Hardening (Weeks 11–13)

**Goal:** Production-ready. Latency minimized. Teams UI changes don't break it.

**Deliverables:**
- Speculative pre-generation (Option B from ADR-001): LLM call starts on VAD speech-start, not speech-end
- Adaptive review timeout based on `conversation_phase`
- Selector health monitor: weekly automated test against live Teams to detect broken selectors
- Structured logging with `call_id` + `turn_number` across all modules
- One-command installer for virtual devices (Linux + Windows)
- Performance dashboard: latency percentiles per pipeline stage, per session

**Exit criteria:**
- P50 turn latency (end-of-utterance → first audio byte of response) < 5 seconds with speculative pre-gen
- P95 turn latency < 10 seconds
- Browser automation success rate > 95% across 50 test runs
- System runs stably for a 30-minute continuous call with no memory leaks or audio drift

---

## Cross-Cutting Concerns

### Logging standard

Every module writes to a shared structured log using this schema:

```python
@dataclass
class LogEvent:
    timestamp_ms: int       # epoch ms
    call_id: str            # uuid, shared across all modules for this call
    turn_number: int
    module: str             # "stt", "llm", "tts", "browser", "orchestrator", "ui"
    event_type: str         # e.g., "utterance_complete", "options_generated"
    duration_ms: int | None # for timed operations
    payload: dict           # event-specific data
    level: str              # "info", "warn", "error"
```

Log to `logs/call_{call_id}.jsonl`. Rotate on call end. Keep last 20 calls.

### Hardware requirements (minimum for v1 with LipSync disabled)

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3060 (12GB VRAM) | RTX 3090 (24GB VRAM) |
| CPU | 8-core, 3GHz | 12-core, 4GHz |
| RAM | 16GB | 32GB |
| Storage | 20GB free (models) | 50GB |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| Network | Stable broadband | Stable broadband |

With LipSync enabled (v2), minimum GPU becomes RTX 3080 (10GB VRAM).

### Terms of Service and Ethics

The following must be documented in the project README and accepted by the user at first run:

1. This tool automates a browser session using your own credentials. Automating Teams may violate Microsoft's Terms of Service. Use at your own risk.
2. Voice cloning synthesizes your own voice. Never use this system to impersonate anyone other than yourself.
3. The other party on the call does not know they are interacting with an AI-assisted system. Consider whether disclosure is appropriate in your context.
4. Recording and transcribing calls may require all-party consent in your jurisdiction. Check local laws before use.

A first-run consent dialog is required before any call can be initiated.

### Monitoring what will break

Three things will break regularly and need active maintenance:

1. **Teams selectors** (monthly): Set up a weekly headless smoke test that logs into Teams and verifies the top-level navigation elements are findable. Alert on failure.
2. **ElevenLabs API changes** (quarterly): Pin to a specific API version. Subscribe to their changelog.
3. **Whisper model updates** (optional): faster-whisper releases new models. Test before upgrading — accuracy improvements can shift VAD timing.

---

## Consequences

### What this plan enables

- A working MVP (Phase 3) within 7 weeks that delivers the core value proposition with no GPU requirement (using Google STT + ElevenLabs).
- A modular architecture where each phase builds cleanly on the last — no phase requires rework of previous phases.
- A system that degrades gracefully: every module has a fallback, and MANUAL_OVERRIDE is always one keystroke away.
- Detailed session logs enable post-call debugging and prompt iteration without needing to reproduce failures live.

### What remains hard

- Browser automation will always be a maintenance liability. Expect to spend ~2 hours/month updating selectors as Teams evolves.
- The latency floor is real. Even with speculative pre-generation, the principal will sometimes experience 6–8 seconds of "hmm..." before a response plays. This is inherent to the human review requirement.
- Voice cloning quality is accent- and mic-dependent. The setup guide must include clear instructions for recording the voice sample (quiet room, good mic, natural speech at normal pace).

### What to revisit after Phase 3 (MVP)

- Should the review timeout be adaptive? (Start at 8s for new users, shrink as they become experienced.)
- Is the 4-option format right? Some principals may prefer 2 high-quality options over 4 mixed-quality options.
- Should the LLM generate a confidence score per option, and use it to adjust font weight in the UI?
- After real usage data: which `conversation_phase` values see the most manual overrides? Those phases need better prompt tuning.

---

## Action Items (ordered by phase)

**Before Phase 1:**
- [ ] Procure or confirm access to RTX 3090 (or equivalent) development machine
- [ ] Confirm legal review of ToS implications for target use context
- [ ] Create Teams test account for automation development (don't use primary account for testing)
- [ ] Record 30s voice sample for TTS cloning

**Phase 1:**
- [ ] Implement `BrowserModule` with three-tier selector strategy
- [ ] Write pre-flight auth check
- [ ] Implement chat history scraper → structured JSON
- [ ] Write virtual device setup script for Linux

**Phase 2:**
- [ ] Integrate silero-vad for end-of-utterance detection
- [ ] Implement faster-whisper sliding window for partial transcripts
- [ ] Set up ElevenLabs WebSocket streaming pipeline
- [ ] Validate audio routing: no echo, no feedback loop

**Phase 3 (MVP):**
- [ ] Implement two-temperature LLM generation
- [ ] Implement `repair_llm_response()` JSON repair function
- [ ] Implement rolling window + incremental summarization
- [ ] Build React control UI with WebSocket bridge
- [ ] Pre-synthesize 20-response audio cache
- [ ] Implement filler deque (no repetition within 5 turns)
- [ ] Write full state machine with all 11 states

**Phase 4:**
- [ ] Benchmark Wav2Lip chunked pipeline on target GPU
- [ ] Implement static face fallback with automatic activation

**Phase 5:**
- [ ] Add barge-in detection (`speech_detected_during_playback`)
- [ ] Add BACKGROUND_ACTION state
- [ ] Implement post-call summarization with mission-completion assessment
- [ ] Build session replay tool

**Phase 6:**
- [ ] Implement speculative pre-generation
- [ ] Implement adaptive review timeout
- [ ] Set up weekly selector smoke test
- [ ] Build per-stage latency dashboard

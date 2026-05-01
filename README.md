# Avatar Agent

An AI agent that conducts live voice calls on your behalf. It listens to the other party via speech-to-text, generates response candidates using an LLM, lets you review and approve responses within a configurable window, then speaks the response via TTS — all while displaying a lip-synced avatar of your face.

**Current status: Phase 1 — Browser Automation Foundation**

Phase 1 delivers: open a Teams call, read chat history for context, and start the call. No audio or LLM responses during the call yet — that comes in Phase 2–3.

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An [Anthropic API key](https://console.anthropic.com/) for LLM planning
- Microsoft Teams account (web version)
- [Playwright](https://playwright.dev/) Chromium browser

**Optional (for later phases):**
- NVIDIA GPU with CUDA (for local Whisper STT and Wav2Lip)
- [ElevenLabs API key](https://elevenlabs.io/) for voice synthesis
- VB-Cable (Windows) or BlackHole (macOS) for virtual audio

---

## Quick Start

```bash
# Install dependencies
cd engine
pip install -e ".[dev]"
playwright install chromium

# Set your API key
export ANTHROPIC_API_KEY="your-key-here"

# Generate a plan (dry run — no browser launched)
python avatar.py plan "Call Alex on Teams to schedule a Sunday meeting" --dry-run

# Generate and execute a plan (opens a real browser)
python avatar.py plan "Call Alex on Teams to schedule a Sunday meeting"

# Check engine status
python avatar.py health
```

---

## Project Structure

```
avatar-agent/
├── avatar.py               # CLI entry point
├── engine/
│   ├── config.yaml         # Configuration (edit this)
│   ├── profiles/
│   │   └── default.json    # Your principal profile (edit this)
│   ├── orchestrator/
│   │   └── state_machine.py  # Central state machine
│   ├── modules/
│   │   ├── browser/        # Playwright browser automation
│   │   ├── llm/            # Claude API + planning
│   │   ├── audio/          # STT + TTS (Phase 2+)
│   │   └── lipsync/        # Wav2Lip video (Phase 4+)
│   └── tests/
└── docs/
    ├── ADR-001-*.md        # Architecture evaluation
    └── ADR-002-*.md        # Production build plan
```

---

## Configuration

Edit `engine/config.yaml` and `engine/profiles/default.json` before running.

**Required settings:**
```yaml
llm:
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"   # set via env var
```

**Your principal profile** (`engine/profiles/default.json`):
Fill in your name, role, tone, and topics to avoid. This shapes how the AI responds on your behalf.

---

## Phase Roadmap

| Phase | Status | What it adds |
|-------|--------|-------------|
| 1 — Browser Foundation | ✅ **Done** | Open Teams, find contacts, start calls |
| 2 — Audio Pipeline | 🔜 | Capture call audio, real-time STT, inject TTS |
| 3 — LLM + HITL | 🔜 | Generate response options, review UI |
| 4 — TTS + Full Loop | 🔜 | Speak responses, full autonomous loop |
| 5 — Mobile App | 🔜 | Monitor and control from your phone |
| 6 — LipSync + Video | 🔜 | Lip-synced avatar video |

---

## Running Tests

```bash
cd engine
pytest tests/ -v
```

---

## Ethics & Terms of Service

By using this software you acknowledge:

1. **Teams ToS**: This tool automates a browser session using your own credentials. Automating Microsoft Teams may violate their Terms of Service. Use at your own risk.

2. **Voice cloning**: The TTS voice synthesis uses your own voice. Never use this system to impersonate anyone other than yourself.

3. **Disclosure**: The other party on the call does not know they are interacting with an AI-assisted system. Consider whether disclosure is appropriate in your context and jurisdiction.

4. **Recording consent**: Transcribing calls may require all-party consent in your jurisdiction. Check your local laws before use.

A first-run consent dialog will be required before any call can be initiated (coming in Phase 3).

---

## Architecture

See the Architecture Decision Records in `docs/`:
- [ADR-001](docs/ADR-001-AI-Avatar-Agent-Architecture.md) — Architecture evaluation and risk analysis
- [ADR-002](docs/ADR-002-Production-Build-Plan.md) — Production build plan (6 phases, 13 weeks)

---

## License

See LICENSE file.

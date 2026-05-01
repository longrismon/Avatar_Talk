# ADR-001: AI Avatar Agent — Architecture Evaluation

**Status:** Proposed (Awaiting Review)
**Date:** 2026-04-15
**Deciders:** Principal Engineer, Product Owner, Legal Counsel (required — see Risks section)
**Document type:** Architecture Decision Record + Design Evaluation

---

## Context

This document evaluates the proposed AI Avatar Agent system, which enables an LLM to conduct live voice calls on behalf of a user (the "principal") by generating response candidates that the principal reviews before they are synthesized via TTS and injected into a real-time call via a virtual microphone and camera. The system also uses browser automation (Playwright) to navigate communication apps (Teams, Slack) and initiate calls.

The core loop is: **STT → LLM response generation → Human review (5s window) → TTS → LipSync → virtual mic/cam injection**, orchestrated by an 11-state finite state machine.

This is a genuinely novel architecture. The evaluation below is structured around the five highest-leverage decision points, followed by cross-cutting concerns.

---

## Decision 1: End-to-End Latency Budget

### The Problem

The most critical risk in this system is **conversation-killing silence gaps**. Every turn in the call incurs:

| Stage | Typical Duration | Worst Case |
|---|---|---|
| VAD silence threshold (end-of-utterance detection) | 1.5s | 1.5s (fixed) |
| Filler TTS synthesis + playback start | 0.3–0.8s | 2s |
| LLM generation (claude-sonnet, ~200 tok output) | 1.5–3s | 8s (timeout) |
| Human review window | 0–5s | 5s (auto-select) |
| TTS first-byte latency (ElevenLabs streaming, level 3) | 0.3–0.5s | 2s |

**Best case end-to-end gap** (principal selects instantly, LLM is fast): ~3.5s of silence before the response plays.

**Realistic case** (principal reads options, moderate LLM latency): ~8–12s of dead air, partially covered by a filler phrase.

In natural conversation, more than ~3 seconds of silence is socially disruptive. The filler phrase buys some cover, but a single "Hmm, let me think..." filler phrase will sound odd if the response doesn't arrive quickly after it.

### Options Considered

**Option A: Current design — sequential pipeline (proposed)**
The filler plays, options generate, principal reviews, TTS plays.

Pros: Simple mental model, clear separation of concerns.

Cons: Cumulative latency is the sum of every stage. The 5-second review window is the dominant term and cannot be reduced below ~2s without making the principal feel rushed.

**Option B: Speculative pre-generation**
Begin LLM generation immediately when VAD detects speech *starting* (not ending), using partial real-time transcription as input. Re-generate or patch the options when the utterance finalizes.

Pros: LLM generation runs in parallel with the last 2–4 seconds of the other party's speech, potentially eliminating most generation latency.

Cons: Requires a streaming STT → streaming LLM pipeline. The LLM output may need to be discarded/regenerated if the utterance takes an unexpected turn (e.g., the other party asks a very long question). Adds significant complexity.

**Option C: Adaptive timeout based on conversation phase**
During casual/social phases (call opening, wrap-up), use a longer review window (10–15s). During high-tempo exchanges (scheduling, quick back-and-forth), auto-reduce to 2–3s and bias toward recommending the most "safe" option.

Pros: Reduces pressure during critical moments without compromising quality during casual moments. Requires conversation-phase awareness, which the LLM already provides via the `conversation_phase` meta field.

### Recommendation

**Adopt Option B (speculative pre-generation) as a V2 target; ship Option C as a V1 improvement over the current design.** Specifically:
- Use the `conversation_phase` field the LLM already returns to dynamically adjust the review timeout — short during `negotiation` and `closing`, longer during `rapport_building` and `opening`.
- Add a second filler phrase slot to the `meta` output: `suggested_filler_2`, played if the principal hasn't selected within 3 seconds, before auto-select fires.
- Track `avg_selection_time_ms` per user across sessions; use it to personalize the default timeout.

---

## Decision 2: State Machine Completeness

### Gaps and Issues Identified

**Gap 1: No barge-in / interruption detection in SPEAKING state.**
The current design transitions from SPEAKING → LISTENING only when `playback_complete` fires. If the other party starts speaking mid-response (a common behavior in natural conversation), the system doesn't detect it. The virtual mic continues injecting TTS audio over the other party's speech, which is both socially inappropriate and loses their content.

*Recommendation:* Add a `speech_detected_during_playback` event that transitions SPEAKING → LISTENING (abort playback) or SPEAKING → SPEAKING + LISTENING (split-capture mode). The LipSync module should support an "interrupted" animation.

**Gap 2: No mid-call browser action state.**
The mission may require actions during an active call (e.g., "while you're talking, check if there's a calendar conflict"). There is no path from any active-call state into BROWSER_ACTION and back. Currently this would require MANUAL_OVERRIDE + manual browsing.

*Recommendation:* Add a BACKGROUND_ACTION state reachable from LISTENING and MANUAL_OVERRIDE that runs browser actions without interrupting the call audio pipeline. Gate this on the `flag_for_principal` field — only trigger it when the LLM surfaces an action request.

**Gap 3: GENERATING state filler timing race condition.**
The spec says "immediately trigger the TTS to play the suggested_filler from the *previous* turn." On turn 1, there is no previous filler. The system needs a bootstrap default filler set (e.g., "Yeah...", "Right...", "Hmm...") and a mechanism to pick one that is not repetitive.

*Recommendation:* Maintain a `recent_fillers` deque (last 5) and exclude those from the random selection pool. Also ensure the filler TTS request fires before the LLM request is dispatched, not after, since TTS has its own latency.

**Gap 4: Token budget management for long calls.**
Long calls (30+ turns) will accumulate transcripts that exceed the LLM's context window. The spec includes a `chat_history_summary` (pre-summarized) and `call_transcript` (raw) but does not specify a strategy for trimming the transcript as the call grows.

*Recommendation:* Add a rolling window strategy: include only the last N turns in `call_transcript` (configurable, default 10), and maintain a separate `call_summary_so_far` field that the LLM updates every 5 turns using a lightweight summarization pass. This is a significant implementation task — flag it as a required pre-launch item.

**Gap 5: No handling of multi-party calls.**
The state machine assumes a 1:1 call (`speaker: "alex"`). A 3-way call would require per-speaker diarization and a much more complex context model.

*Recommendation:* Document this as an explicit V1 constraint in the configuration schema (`max_participants: 1`). Detect and surface an error if a third party joins mid-call.

---

## Decision 3: Technology Component Choices

### STT: faster-whisper large-v3 vs. Google STT

| Dimension | faster-whisper large-v3 | Google STT |
|---|---|---|
| Latency (streaming) | ~0.5–1.5s on GPU (RTX 3080+) | ~0.3–0.8s |
| Accuracy (English) | Excellent | Excellent |
| Cost | GPU compute only | ~$0.016/min (v1) |
| Privacy | Local — audio stays on device | Cloud — audio sent to Google |
| Offline capability | Yes | No |
| Setup complexity | High (CUDA, model download ~3GB) | Low |

**Assessment:** The `large-v3` model requires a mid-tier GPU to run in real-time. On CPU, transcription will be 5–10x slower than real-time — completely unusable. If the target environment is a laptop without a discrete GPU, the primary should be **Google STT**, not faster-whisper, with faster-whisper as a local fallback when GPU is available. The config should auto-detect and swap.

### TTS: ElevenLabs vs. Coqui XTTS

| Dimension | ElevenLabs (streaming) | Coqui XTTS |
|---|---|---|
| Voice quality | Excellent | Good |
| First-byte latency | 300–500ms (latency_opt: 3) | 800–1500ms |
| Cost | ~$0.30/1K chars; 5-min call ≈ $1.50–$4 | Free (GPU compute) |
| Voice cloning quality | Best-in-class | Good but noticeably synthetic |
| Concurrent calls | Unlimited (API) | GPU-limited |
| Offline | No | Yes |

**Assessment:** ElevenLabs is the correct primary choice for quality. The cost per call is acceptable for the use case. The `latency_optimization: 3` setting is appropriate. However, at level 4, latency drops further at the cost of occasional audio artifacts — worth A/B testing. Coqui XTTS as fallback is reasonable but the quality gap may break the impersonation. Consider using a pre-baked "safe" response set (audio files, not synthesized) as the true fallback for sub-500ms recovery.

### LipSync: Wav2Lip vs. SadTalker vs. MuseTalk

| Dimension | Wav2Lip | SadTalker | MuseTalk |
|---|---|---|---|
| Quality (mouth sync) | Good, but blurry mouth region | Better realism | Best quality |
| Head movement | None (static) | Natural head motion | Moderate |
| Latency (streaming) | Fastest (~25fps possible) | Slow (~1–2fps, offline) | Moderate |
| GPU requirement | RTX 2070+ | RTX 3080+ | RTX 3080+ |
| Real-time capable | Yes (just barely) | No | Partial |

**Assessment:** Wav2Lip is the only viable option for real-time use today. SadTalker and MuseTalk generate offline and are unusable for live calls. The spec's `enable: false` (camera-off mode for v1) is a pragmatic call — camera-off is strongly recommended for v1 to reduce GPU demand, latency, and the uncanny valley risk. A frozen frame or avatar graphic is preferable to a glitchy lip-sync that breaks the illusion. Treat LipSync as a V2 feature.

### Browser Automation: Playwright + Headless=false

The spec uses `headless: false`, which is necessary because:
- Teams and Slack detect headless browsers and may block access
- Virtual camera/mic injection requires the browser to actually have a media session

**Risks:**
- Teams web UI changes frequently (Microsoft ships updates weekly). CSS selectors will break. The system needs a selector abstraction layer (semantic locators + visual fallback via screenshot + LLM description).
- The `grant_permissions` step using xdotool/AppleScript is OS-specific and fragile. Chrome's `--use-fake-device-for-media-stream` and `--allow-file-access-from-files` flags + pre-authorized permission profiles are a more reliable approach.
- Browser authentication: the `user_data_dir` strategy works but 2FA/SSO tokens expire. No state is defined for re-authentication mid-session.

**Recommendation:** Add a `browser_health_check` step at PLANNING time that verifies login state and prompts re-auth before the call starts, not mid-automation.

---

## Decision 4: LLM Prompt Architecture

### Strengths

The context payload design is well-thought-out:
- `principal_profile` is static and loaded once — correct.
- `mission.completed_steps` gives the LLM conversation-level awareness of progress — this will meaningfully steer responses toward the goal.
- `system_state.intervention_history` allows the LLM to learn from this session's manual overrides, which is a subtle but powerful self-correction mechanism.
- The `suggested_filler` field is a genuine latency mitigation, not just decoration.
- `flag_for_principal` as a structured field is the right pattern for surfacing anomalies without disrupting output format.

### Weaknesses and Recommendations

**W1: No token budget enforcement.** A 30-turn call transcript with the full principal profile will approach or exceed 8,000 tokens before response. The LLM `max_tokens: 1000` is for the *output*, not the input. Add `max_context_tokens` to the config and implement truncation before sending.

**W2: The validation logic (exactly 4 options, exactly 1 recommended) should be a deterministic parser, not a re-prompt.** Re-prompting the LLM for a formatting error wastes 2–4 seconds. Instead, implement a JSON repair function that can:
- Add a missing `recommended: false` to options without it
- Promote the first option to recommended if none is flagged
- Truncate a 5th option if present

Reserve re-prompting only for semantic violations (topics_to_avoid, identity disclosure).

**W3: Temperature 0.7 may be too high for safety-critical selections.** The recommended option is auto-selected if the principal doesn't intervene. At temperature 0.7, the system has meaningful variance in what it recommends. Consider a two-temperature strategy: generate options at 0.7 for diversity, then run a 0-temperature selection pass to deterministically pick the recommended option based on mission alignment.

**W4: No explicit handling of the other party speaking a language other than the configured `language: "en"`.** If Alex switches to another language, the STT will transcribe poorly, the LLM will get garbled input, and the generated responses will be in English. Add a language-detection step in LISTENING and surface a flag if the detected language differs from the configured language.

---

## Decision 5: Ethical, Legal, and Detection Risk

This section cannot be omitted from a proper architecture review.

### Legal Risk: Identity Deception

The system's core value proposition requires impersonating a real person in a live conversation without the other party's knowledge or consent. This is legally complex:

- In many jurisdictions, recording a conversation without all-party consent is illegal (e.g., California, Germany, UK under some interpretations of GDPR).
- The system does not record audio of the other party for playback — it transcribes it — but capturing and processing voice data without consent may still constitute a GDPR/CCPA violation depending on interpretation.
- If the other party is deceived into making a commitment (scheduling a meeting, agreeing to a contract term) while believing they are speaking with a human, this may constitute fraud in some contexts.
- Several countries have introduced or are introducing AI disclosure laws (EU AI Act, proposed US DEFIANCE Act) that may require disclosure when an AI is conducting a conversation.

**Recommendation:** Require legal review before production deployment. At minimum, define a clear acceptable-use policy. Consider adding a consent mechanism option: a brief disclosure at the start of the call ("I'm using an AI assistant to help with this call") before the agent takes over.

### Detection Risk

The illusion can be broken by:
- **Response timing patterns**: A consistent ~8–12 second response time will be noticed in longer calls.
- **Unnatural filler repetition**: Saying "Hmm, let me think..." every single turn is conspicuous.
- **Knowledge gaps**: If Alex asks "Hey, did you see my email from yesterday?" and the system has no email context, the deflection options may sound evasive.
- **Lip-sync artifacts**: Wav2Lip's blurry mouth region is distinctive to people familiar with deepfake artifacts.

**Recommendation:** Expand `principal_profile.known_facts` to include an email/calendar context field, populated automatically by integrating with the user's email/calendar before the call starts (as an optional BROWSER_ACTION step in planning).

---

## Trade-off Summary

| Dimension | Current Design | With Recommendations |
|---|---|---|
| Latency (best case) | ~3.5s silence gap | ~2s with speculative pre-gen (V2) |
| Latency (typical) | ~8–12s silence gap | ~5–7s with adaptive timeout (V1) |
| Context window management | None | Rolling window + per-5-turn summarization |
| LipSync | Real-time (Wav2Lip, v1) | Disabled in v1, Wav2Lip in v2 |
| Barge-in handling | Not supported | Add speech_detected_during_playback event |
| Multi-party calls | Not handled | Blocked with explicit error |
| Browser auth expiry | Not handled | Pre-flight health check |
| Legal compliance | Not specified | Requires legal review + consent option |

---

## Consequences

### What becomes easier with this architecture

- The human-in-the-loop design (HUMAN_REVIEW state with keyboard shortcuts) makes this meaningfully safer than a fully autonomous agent. The principal can always intervene.
- The MANUAL_OVERRIDE state with STT-continued transcription means the conversation log stays complete even when the principal takes over, which is valuable for post-call summarization.
- The structured LLM output (JSON with `strategy`, `tone`, `reasoning`) gives the principal genuine signal for selecting between options quickly, not just text blobs.
- The fallback chains (ElevenLabs → Coqui, faster-whisper → Google, LLM retry → cached set) mean no single component failure crashes the call.

### What becomes harder

- The system is deeply multi-process: browser, STT engine, LLM API, TTS API, LipSync, virtual device driver, and control UI must all operate concurrently with tight latency budgets. Process supervision (systemd, PM2, or a custom watchdog) is essential.
- Debugging a failed call requires correlating logs from 5–6 separate subsystems with millisecond timestamps. Structured logging with a shared `call_id` and `turn_number` field across all components is required from day 1.
- Each new communication platform (Zoom, Google Meet, Slack) requires its own Playwright selector set and permission-granting logic. This creates ongoing maintenance overhead.

### What must be revisited before v1.0

- GPU requirements for the target environment must be documented and validated. If GPU is not guaranteed, the STT primary must change.
- Legal review of the impersonation mechanism is required.
- Token budget management in GENERATING must be implemented.
- Barge-in detection in SPEAKING must be added.
- A filler diversity mechanism must prevent repetitive fillers.

---

## Action Items

1. [ ] **[Blocking]** Legal review: assess identity deception risk in target jurisdictions and define acceptable use policy
2. [ ] **[Blocking]** Add token budget enforcement in GENERATING state: implement `max_context_tokens` config + rolling window truncation
3. [ ] **[Blocking]** Add barge-in detection event `speech_detected_during_playback` in SPEAKING state
4. [ ] **[V1 improvement]** Implement adaptive review timeout based on `conversation_phase` LLM meta field
5. [ ] **[V1 improvement]** Add `recent_fillers` deque to prevent filler repetition; add `suggested_filler_2` to LLM output schema
6. [ ] **[V1 improvement]** Add browser pre-flight `browser_health_check` step in PLANNING to verify authentication state
7. [ ] **[V1 improvement]** Add multi-party detection: error if third participant joins mid-call
8. [ ] **[V1 improvement]** Add JSON repair function in GENERATING for formatting violations; reserve re-prompt for semantic violations only
9. [ ] **[V2]** Implement speculative pre-generation: begin LLM call when VAD detects speech *start*, using streaming partial transcript
10. [ ] **[V2]** Add BACKGROUND_ACTION state for mid-call browser operations (calendar check, email lookup)
11. [ ] **[V2]** Add per-5-turn conversation summarization in GENERATING to maintain long-call context quality
12. [ ] **[Decision needed]** Camera-off mode: strongly recommend disabling LipSync for v1 (set `enable: false`). Confirm this is acceptable to stakeholders.
13. [ ] **[Decision needed]** GPU availability: document minimum hardware spec. If no GPU guaranteed, switch STT primary to Google STT.
14. [ ] **[Decision needed]** Consider two-temperature LLM strategy: 0.7 for option generation, 0.0 for recommended selection.

---

## Appendix: Revised State Transition Table (with Recommendations)

| From | Event | To |
|---|---|---|
| IDLE | instruction_received | PLANNING |
| PLANNING | browser_health_check_failed | ERROR |
| PLANNING | plan_confirmed | BROWSER_ACTION |
| PLANNING | plan_rejected | IDLE |
| PLANNING | timeout / error | ERROR |
| BROWSER_ACTION | step_succeeded (more) | BROWSER_ACTION |
| BROWSER_ACTION | call_initiated | AWAITING_CALL |
| BROWSER_ACTION | step_failed | ERROR |
| AWAITING_CALL | call_connected | LISTENING |
| AWAITING_CALL | declined / timeout | ERROR |
| AWAITING_CALL | third_party_detected | ERROR |
| LISTENING | utterance_complete | GENERATING |
| LISTENING | call_ended | CALL_ENDED |
| LISTENING | user_override | MANUAL_OVERRIDE |
| LISTENING | background_action_triggered | BACKGROUND_ACTION *(new)* |
| GENERATING | options_ready | HUMAN_REVIEW |
| GENERATING | generation_failed | ERROR |
| HUMAN_REVIEW | option_selected / timeout | SPEAKING |
| HUMAN_REVIEW | user_takeover | MANUAL_OVERRIDE |
| SPEAKING | playback_complete | LISTENING |
| SPEAKING | **speech_detected_during_playback** *(new)* | LISTENING |
| SPEAKING | user_interrupt | MANUAL_OVERRIDE |
| SPEAKING | call_ended | CALL_ENDED |
| MANUAL_OVERRIDE | resume_ai | LISTENING |
| MANUAL_OVERRIDE | call_ended | CALL_ENDED |
| BACKGROUND_ACTION *(new)* | action_complete | LISTENING |
| BACKGROUND_ACTION *(new)* | action_failed | MANUAL_OVERRIDE |
| CALL_ENDED | dismissed | IDLE |
| ERROR | resolved | (originating) |
| ERROR | user_abort | IDLE |
| ERROR | user_takeover | MANUAL_OVERRIDE |

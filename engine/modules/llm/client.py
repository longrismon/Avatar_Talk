"""LLM client implementations — Anthropic, OpenAI, Ollama."""
import json
import os
import re
from pathlib import Path
from typing import Optional

import httpx

from .interface import ActionPlan, LLMClient, ResponseOptions

# Load planning prompt once at module import
_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt file by name (without .txt extension)."""
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def repair_llm_response(raw: str) -> dict:
    """
    Parse and repair common LLM JSON formatting errors.

    Handles:
    - Markdown code fences (```json ... ```)
    - Trailing commas before ] or }
    - Missing 'recommended' field in options
    - More than 4 options (truncates to 4)
    - No option marked recommended (promotes first)

    Raises:
        json.JSONDecodeError: If the response cannot be parsed even after repair.
    """
    # Strip markdown fences
    text = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text.strip(), flags=re.MULTILINE)
    text = text.strip()

    # Fix trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)

    data = json.loads(text)  # raises json.JSONDecodeError if still invalid

    # Structural repairs for response options (Phase 3+)
    if "options" in data and isinstance(data["options"], list):
        # Truncate to 4 options
        if len(data["options"]) > 4:
            data["options"] = data["options"][:4]
        # Ensure recommended field exists on all options
        for opt in data["options"]:
            if "recommended" not in opt:
                opt["recommended"] = False
        # Promote first option if none is recommended
        if not any(opt.get("recommended") for opt in data["options"]):
            data["options"][0]["recommended"] = True

    return data


_REC_SYSTEM = (
    "Given these response options and the conversation goal, respond with only a JSON "
    "object: {\"recommended_id\": <1-4>}. Pick the option that best advances the goal."
)


class AnthropicLLM(LLMClient):
    """Claude API client via Anthropic's Messages API."""

    API_URL = "https://api.anthropic.com/v1/messages"
    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", timeout: float = 15.0):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._planning_prompt: Optional[str] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": self.ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                timeout=self._timeout,
            )
        return self._client

    def _get_planning_prompt(self) -> str:
        if self._planning_prompt is None:
            self._planning_prompt = _load_prompt("planning")
        return self._planning_prompt

    async def _chat(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        """Single Anthropic Messages API call; returns the assistant text."""
        response = await self._get_client().post(
            self.API_URL,
            json={
                "model": self._model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]

    async def generate_plan(self, instruction: str, available_apps: list[str]) -> ActionPlan:
        """Call Claude to decompose instruction into action steps."""
        payload = {
            "instruction": instruction,
            "available_apps": available_apps,
        }
        raw_text = await self._chat(
            system=self._get_planning_prompt(),
            user=json.dumps(payload),
            temperature=0.3,
            max_tokens=1000,
        )
        parsed = repair_llm_response(raw_text)
        return ActionPlan(
            steps=parsed.get("steps", []),
            mission_summary=parsed.get("mission_summary", ""),
            estimated_duration=parsed.get("estimated_duration", ""),
            conversation_goal=parsed.get("conversation_goal", ""),
            success_criteria=parsed.get("success_criteria", ""),
        )

    async def generate_responses(self, context_payload: dict) -> ResponseOptions:
        """Two-temperature response generation: diversity pass + deterministic recommendation."""
        # Call 1 — diversity pass
        raw_options = await self._chat(
            system=_load_prompt("conversation"),
            user=json.dumps(context_payload),
            temperature=0.7,
            max_tokens=800,
        )
        data = repair_llm_response(raw_options)

        # Call 2 — deterministic recommendation
        rec_payload = {
            "options": data["options"],
            "mission_goal": context_payload.get("mission_goal", ""),
            "current_utterance": context_payload.get("current_utterance", ""),
        }
        try:
            raw_rec = await self._chat(
                system=_REC_SYSTEM,
                user=json.dumps(rec_payload),
                temperature=0.0,
                max_tokens=50,
            )
            rec = json.loads(raw_rec)
            rec_id = int(rec.get("recommended_id", 1))
            for opt in data["options"]:
                opt["recommended"] = (opt["id"] == rec_id)
            # Ensure exactly one recommended
            if not any(opt["recommended"] for opt in data["options"]):
                data["options"][0]["recommended"] = True
        except Exception:
            pass  # repair_llm_response already ensured one recommended=True

        return ResponseOptions(options=data["options"])

    async def summarize_call(self, transcript: list[dict], mission: dict) -> dict:
        """Summarize the call so far into 2-3 sentences."""
        payload = {
            "mission_goal": mission.get("conversation_goal", ""),
            "turns": transcript,
            "previous_summary": mission.get("summary", ""),
        }
        raw = await self._chat(
            system=_load_prompt("summary"),
            user=json.dumps(payload),
            temperature=0.0,
            max_tokens=200,
        )
        try:
            result = json.loads(raw)
            return {"summary": result.get("summary", raw)}
        except Exception:
            return {"summary": raw.strip()}

    async def aclose(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class OllamaLLM(LLMClient):
    """Local LLM via Ollama (offline fallback)."""

    def __init__(self, model: str = "llama3:8b-instruct", base_url: str = "http://localhost:11434", timeout: float = 30.0):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    async def _chat(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        response = await self._get_client().post(
            "/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def generate_plan(self, instruction: str, available_apps: list[str]) -> ActionPlan:
        payload = {
            "instruction": instruction,
            "available_apps": available_apps,
        }
        raw_text = await self._chat(
            system=_load_prompt("planning"),
            user=json.dumps(payload),
            temperature=0.3,
            max_tokens=1000,
        )
        parsed = repair_llm_response(raw_text)
        return ActionPlan(
            steps=parsed.get("steps", []),
            mission_summary=parsed.get("mission_summary", ""),
            estimated_duration=parsed.get("estimated_duration", ""),
            conversation_goal=parsed.get("conversation_goal", ""),
            success_criteria=parsed.get("success_criteria", ""),
        )

    async def generate_responses(self, context_payload: dict) -> ResponseOptions:
        raw_options = await self._chat(
            system=_load_prompt("conversation"),
            user=json.dumps(context_payload),
            temperature=0.7,
            max_tokens=800,
        )
        data = repair_llm_response(raw_options)

        rec_payload = {
            "options": data["options"],
            "mission_goal": context_payload.get("mission_goal", ""),
            "current_utterance": context_payload.get("current_utterance", ""),
        }
        try:
            raw_rec = await self._chat(
                system=_REC_SYSTEM,
                user=json.dumps(rec_payload),
                temperature=0.0,
                max_tokens=50,
            )
            rec = json.loads(raw_rec)
            rec_id = int(rec.get("recommended_id", 1))
            for opt in data["options"]:
                opt["recommended"] = (opt["id"] == rec_id)
            if not any(opt["recommended"] for opt in data["options"]):
                data["options"][0]["recommended"] = True
        except Exception:
            pass

        return ResponseOptions(options=data["options"])

    async def summarize_call(self, transcript: list[dict], mission: dict) -> dict:
        payload = {
            "mission_goal": mission.get("conversation_goal", ""),
            "turns": transcript,
            "previous_summary": mission.get("summary", ""),
        }
        raw = await self._chat(
            system=_load_prompt("summary"),
            user=json.dumps(payload),
            temperature=0.0,
            max_tokens=200,
        )
        try:
            result = json.loads(raw)
            return {"summary": result.get("summary", raw)}
        except Exception:
            return {"summary": raw.strip()}

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class CustomLLM(LLMClient):
    """Generic OpenAI-compatible chat completions client (custom BASE_URL/API_KEY/MODEL)."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 15.0):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._planning_prompt: Optional[str] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"content-type": "application/json"}
            if self._api_key:
                headers["authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=self._timeout,
            )
        return self._client

    def _get_planning_prompt(self) -> str:
        if self._planning_prompt is None:
            self._planning_prompt = _load_prompt("planning")
        return self._planning_prompt

    async def _chat(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        response = await self._get_client().post(
            "/chat/completions",
            json={
                "model": self._model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def generate_plan(self, instruction: str, available_apps: list[str]) -> ActionPlan:
        payload = {"instruction": instruction, "available_apps": available_apps}
        raw_text = await self._chat(
            system=self._get_planning_prompt(),
            user=json.dumps(payload),
            temperature=0.3,
            max_tokens=1000,
        )
        parsed = repair_llm_response(raw_text)
        return ActionPlan(
            steps=parsed.get("steps", []),
            mission_summary=parsed.get("mission_summary", ""),
            estimated_duration=parsed.get("estimated_duration", ""),
            conversation_goal=parsed.get("conversation_goal", ""),
            success_criteria=parsed.get("success_criteria", ""),
        )

    async def generate_responses(self, context_payload: dict) -> ResponseOptions:
        raw_options = await self._chat(
            system=_load_prompt("conversation"),
            user=json.dumps(context_payload),
            temperature=0.7,
            max_tokens=800,
        )
        data = repair_llm_response(raw_options)

        rec_payload = {
            "options": data["options"],
            "mission_goal": context_payload.get("mission_goal", ""),
            "current_utterance": context_payload.get("current_utterance", ""),
        }
        try:
            raw_rec = await self._chat(
                system=_REC_SYSTEM,
                user=json.dumps(rec_payload),
                temperature=0.0,
                max_tokens=50,
            )
            rec = json.loads(raw_rec)
            rec_id = int(rec.get("recommended_id", 1))
            for opt in data["options"]:
                opt["recommended"] = (opt["id"] == rec_id)
            if not any(opt["recommended"] for opt in data["options"]):
                data["options"][0]["recommended"] = True
        except Exception:
            pass

        return ResponseOptions(options=data["options"])

    async def summarize_call(self, transcript: list[dict], mission: dict) -> dict:
        payload = {
            "mission_goal": mission.get("conversation_goal", ""),
            "turns": transcript,
            "previous_summary": mission.get("summary", ""),
        }
        raw = await self._chat(
            system=_load_prompt("summary"),
            user=json.dumps(payload),
            temperature=0.0,
            max_tokens=200,
        )
        try:
            result = json.loads(raw)
            return {"summary": result.get("summary", raw)}
        except Exception:
            return {"summary": raw.strip()}

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


def create_llm_client(config) -> LLMClient:
    """Factory: create the primary LLM client from config, fall back if needed.

    Args:
        config: LLMConfig from engine/config.py

    Returns:
        A LLMClient instance (AnthropicLLM or OllamaLLM).
    """
    if config.primary == "anthropic":
        api_key = config.anthropic.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        return AnthropicLLM(
            api_key=api_key,
            model=config.anthropic.model,
            timeout=config.anthropic.timeout_seconds,
        )
    elif config.primary == "ollama":
        return OllamaLLM(
            model=config.ollama.model,
            base_url=config.ollama.base_url,
            timeout=config.ollama.timeout_seconds,
        )
    elif config.primary == "custom":
        return CustomLLM(
            base_url=config.custom.base_url or os.environ.get("CUSTOM_BASE_URL", ""),
            api_key=config.custom.api_key or os.environ.get("CUSTOM_API_KEY", ""),
            model=config.custom.model or os.environ.get("CUSTOM_MODEL", ""),
            timeout=config.custom.timeout_seconds,
        )
    else:
        raise ValueError(f"Unsupported LLM primary: {config.primary}")

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

    async def generate_plan(self, instruction: str, available_apps: list[str]) -> ActionPlan:
        """Call Claude to decompose instruction into action steps."""
        payload = {
            "instruction": instruction,
            "available_apps": available_apps,
        }
        response = await self._get_client().post(
            self.API_URL,
            json={
                "model": self._model,
                "max_tokens": 1000,
                "temperature": 0.3,
                "system": self._get_planning_prompt(),
                "messages": [
                    {"role": "user", "content": json.dumps(payload)}
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        raw_text = data["content"][0]["text"]
        parsed = repair_llm_response(raw_text)

        return ActionPlan(
            steps=parsed.get("steps", []),
            mission_summary=parsed.get("mission_summary", ""),
            estimated_duration=parsed.get("estimated_duration", ""),
            conversation_goal=parsed.get("conversation_goal", ""),
            success_criteria=parsed.get("success_criteria", ""),
        )

    async def generate_responses(self, context_payload: dict) -> ResponseOptions:
        raise NotImplementedError("generate_responses is implemented in Phase 3")

    async def summarize_call(self, transcript: list[dict], mission: dict) -> dict:
        raise NotImplementedError("summarize_call is implemented in Phase 5")

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

    async def generate_plan(self, instruction: str, available_apps: list[str]) -> ActionPlan:
        payload = {
            "instruction": instruction,
            "available_apps": available_apps,
        }
        planning_prompt = _load_prompt("planning")
        response = await self._get_client().post(
            "/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": planning_prompt},
                    {"role": "user", "content": json.dumps(payload)},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        raw_text = data["message"]["content"]
        parsed = repair_llm_response(raw_text)
        return ActionPlan(
            steps=parsed.get("steps", []),
            mission_summary=parsed.get("mission_summary", ""),
            estimated_duration=parsed.get("estimated_duration", ""),
            conversation_goal=parsed.get("conversation_goal", ""),
            success_criteria=parsed.get("success_criteria", ""),
        )

    async def generate_responses(self, context_payload: dict) -> ResponseOptions:
        raise NotImplementedError("generate_responses is implemented in Phase 3")

    async def summarize_call(self, transcript: list[dict], mission: dict) -> dict:
        raise NotImplementedError("summarize_call is implemented in Phase 5")

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

    async def generate_plan(self, instruction: str, available_apps: list[str]) -> ActionPlan:
        payload = {"instruction": instruction, "available_apps": available_apps}
        response = await self._get_client().post(
            "/chat/completions",
            json={
                "model": self._model,
                "temperature": 0.3,
                "messages": [
                    {"role": "system", "content": self._get_planning_prompt()},
                    {"role": "user", "content": json.dumps(payload)},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed = repair_llm_response(raw_text)
        return ActionPlan(
            steps=parsed.get("steps", []),
            mission_summary=parsed.get("mission_summary", ""),
            estimated_duration=parsed.get("estimated_duration", ""),
            conversation_goal=parsed.get("conversation_goal", ""),
            success_criteria=parsed.get("success_criteria", ""),
        )

    async def generate_responses(self, context_payload: dict) -> ResponseOptions:
        raise NotImplementedError("generate_responses is implemented in Phase 3")

    async def summarize_call(self, transcript: list[dict], mission: dict) -> dict:
        raise NotImplementedError("summarize_call is implemented in Phase 5")

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

"""LLM interface — abstract base class for all LLM providers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ActionPlan:
    """Result of LLM planning decomposition."""
    steps: list[dict]
    mission_summary: str
    estimated_duration: str = ""
    conversation_goal: str = ""
    success_criteria: str = ""


@dataclass
class ResponseOptions:
    """4 response options generated for a conversation turn."""
    options: list[dict]
    meta: dict = field(default_factory=dict)


class LLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate_plan(self, instruction: str, available_apps: list[str]) -> ActionPlan:
        """Decompose a user instruction into a sequence of browser action steps."""

    @abstractmethod
    async def generate_responses(self, context_payload: dict) -> ResponseOptions:
        """Generate 4 response options for a conversation turn. (Phase 3)"""

    @abstractmethod
    async def summarize_call(self, transcript: list[dict], mission: dict) -> dict:
        """Generate a post-call summary with action items. (Phase 5)"""

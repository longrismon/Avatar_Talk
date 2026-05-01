"""Tests for LLM planning module — repair_llm_response and generate_plan."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from engine.modules.llm.client import repair_llm_response, AnthropicLLM
from engine.modules.llm.interface import ActionPlan


class TestRepairLlmResponse:
    def test_valid_json_passthrough(self):
        data = {"steps": [], "mission_summary": "Test"}
        result = repair_llm_response(json.dumps(data))
        assert result == data

    def test_strips_markdown_fence(self):
        raw = '```json\n{"steps": [], "mission_summary": "Test"}\n```'
        result = repair_llm_response(raw)
        assert result["mission_summary"] == "Test"

    def test_strips_json_fence_no_language(self):
        raw = '```\n{"steps": [], "mission_summary": "Test"}\n```'
        result = repair_llm_response(raw)
        assert result["mission_summary"] == "Test"

    def test_fixes_trailing_comma_in_object(self):
        raw = '{"steps": [], "mission_summary": "Test",}'
        result = repair_llm_response(raw)
        assert result["mission_summary"] == "Test"

    def test_fixes_trailing_comma_in_array(self):
        raw = '{"steps": [{"a": 1},], "mission_summary": "Test"}'
        result = repair_llm_response(raw)
        assert len(result["steps"]) == 1

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            repair_llm_response("not json at all {{{")

    def test_truncates_options_to_4(self):
        data = {
            "options": [
                {"id": i, "text": f"Option {i}", "recommended": False}
                for i in range(1, 6)  # 5 options
            ]
        }
        result = repair_llm_response(json.dumps(data))
        assert len(result["options"]) == 4

    def test_adds_missing_recommended_field(self):
        data = {
            "options": [
                {"id": 1, "text": "Option 1"},  # missing recommended
                {"id": 2, "text": "Option 2", "recommended": False},
            ]
        }
        result = repair_llm_response(json.dumps(data))
        assert "recommended" in result["options"][0]

    def test_promotes_first_option_if_none_recommended(self):
        data = {
            "options": [
                {"id": 1, "text": "Option 1", "recommended": False},
                {"id": 2, "text": "Option 2", "recommended": False},
            ]
        }
        result = repair_llm_response(json.dumps(data))
        assert result["options"][0]["recommended"] is True

    def test_does_not_change_existing_recommended(self):
        data = {
            "options": [
                {"id": 1, "text": "Option 1", "recommended": False},
                {"id": 2, "text": "Option 2", "recommended": True},
            ]
        }
        result = repair_llm_response(json.dumps(data))
        assert result["options"][1]["recommended"] is True
        # First should not be promoted since second is recommended
        assert result["options"][0]["recommended"] is False


class TestAnthropicLLMGeneratePlan:
    async def test_generate_plan_returns_action_plan(self):
        """Test generate_plan with a mocked HTTP response."""
        plan_response = {
            "steps": [
                {"action": "open_app", "params": {"app_name": "teams"}},
                {"action": "search_contact", "params": {"name": "Alex"}},
                {"action": "read_chat_history", "params": {"contact": "Alex", "limit": 30}},
                {"action": "start_call", "params": {"contact": "Alex", "video": False}},
                {"action": "grant_permissions", "params": {"mic": True, "camera": False}},
            ],
            "mission_summary": "Call Alex on Teams.",
            "estimated_duration": "3-5 minutes",
            "conversation_goal": "Schedule a meeting.",
            "success_criteria": "Meeting time agreed.",
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"text": json.dumps(plan_response)}]
        }
        mock_response.raise_for_status = MagicMock()

        llm = AnthropicLLM(api_key="test-key")

        with patch.object(llm._get_client(), "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            plan = await llm.generate_plan("Call Alex on Teams", ["teams"])

        assert isinstance(plan, ActionPlan)
        assert len(plan.steps) == 5
        assert plan.steps[0]["action"] == "open_app"
        assert plan.mission_summary == "Call Alex on Teams."

    async def test_generate_plan_handles_markdown_fences(self):
        """Test that markdown-fenced JSON from the LLM is repaired correctly."""
        plan_response = {
            "steps": [{"action": "open_app", "params": {"app_name": "teams"}}],
            "mission_summary": "Test plan.",
            "estimated_duration": "1 minute",
            "conversation_goal": "Test.",
            "success_criteria": "Done.",
        }
        raw = f"```json\n{json.dumps(plan_response)}\n```"

        mock_response = MagicMock()
        mock_response.json.return_value = {"content": [{"text": raw}]}
        mock_response.raise_for_status = MagicMock()

        llm = AnthropicLLM(api_key="test-key")
        with patch.object(llm._get_client(), "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            plan = await llm.generate_plan("Open Teams", ["teams"])

        assert plan.mission_summary == "Test plan."
        assert len(plan.steps) == 1

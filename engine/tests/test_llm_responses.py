"""Tests for Phase 3 LLM response generation — generate_responses() and summarize_call()."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_options_json(options=None) -> str:
    if options is None:
        options = [
            {"id": 1, "text": "Option A", "tone": "professional", "recommended": False},
            {"id": 2, "text": "Option B", "tone": "empathetic", "recommended": False},
            {"id": 3, "text": "Option C", "tone": "direct", "recommended": True},
            {"id": 4, "text": "Option D", "tone": "light", "recommended": False},
        ]
    return json.dumps({"options": options})


def _make_rec_json(rec_id: int = 3) -> str:
    return json.dumps({"recommended_id": rec_id})


@pytest.mark.asyncio
class TestAnthropicGenerateResponses:
    async def test_makes_two_api_calls(self):
        from engine.modules.llm.client import AnthropicLLM

        llm = AnthropicLLM(api_key="k", model="m")
        call_count = 0

        async def fake_chat(system, user, temperature, max_tokens):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_options_json()
            return _make_rec_json()

        llm._chat = fake_chat
        result = await llm.generate_responses({"mission_goal": "test", "current_utterance": "hi"})
        assert call_count == 2

    async def test_returns_four_options(self):
        from engine.modules.llm.client import AnthropicLLM

        llm = AnthropicLLM(api_key="k", model="m")
        call_num = 0

        async def fake_chat(system, user, temperature, max_tokens):
            nonlocal call_num
            call_num += 1
            return _make_options_json() if call_num == 1 else _make_rec_json()

        llm._chat = fake_chat
        result = await llm.generate_responses({})
        assert len(result.options) == 4

    async def test_ensures_one_recommended(self):
        from engine.modules.llm.client import AnthropicLLM

        llm = AnthropicLLM(api_key="k", model="m")
        call_num = 0

        async def fake_chat(system, user, temperature, max_tokens):
            nonlocal call_num
            call_num += 1
            return _make_options_json() if call_num == 1 else _make_rec_json(2)

        llm._chat = fake_chat
        result = await llm.generate_responses({})
        recommended = [o for o in result.options if o.get("recommended")]
        assert len(recommended) == 1
        assert recommended[0]["id"] == 2

    async def test_first_call_uses_high_temperature(self):
        from engine.modules.llm.client import AnthropicLLM

        llm = AnthropicLLM(api_key="k", model="m")
        temps = []

        async def fake_chat(system, user, temperature, max_tokens):
            temps.append(temperature)
            if len(temps) == 1:
                return _make_options_json()
            return _make_rec_json()

        llm._chat = fake_chat
        await llm.generate_responses({})
        assert temps[0] == pytest.approx(0.7)
        assert temps[1] == pytest.approx(0.0)

    async def test_handles_json_repair_markdown_fences(self):
        from engine.modules.llm.client import AnthropicLLM

        llm = AnthropicLLM(api_key="k", model="m")
        call_num = 0

        fenced = "```json\n" + _make_options_json() + "\n```"

        async def fake_chat(system, user, temperature, max_tokens):
            nonlocal call_num
            call_num += 1
            return fenced if call_num == 1 else _make_rec_json()

        llm._chat = fake_chat
        result = await llm.generate_responses({})
        assert len(result.options) == 4

    async def test_second_call_failure_keeps_first_recommended(self):
        from engine.modules.llm.client import AnthropicLLM

        llm = AnthropicLLM(api_key="k", model="m")
        call_num = 0

        async def fake_chat(system, user, temperature, max_tokens):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                return _make_options_json()
            raise Exception("API error on rec call")

        llm._chat = fake_chat
        result = await llm.generate_responses({})
        # repair_llm_response should have set first option as recommended
        assert any(o.get("recommended") for o in result.options)

    async def test_summarize_call_returns_text(self):
        from engine.modules.llm.client import AnthropicLLM

        llm = AnthropicLLM(api_key="k", model="m")
        llm._chat = AsyncMock(return_value=json.dumps({"summary": "Alex agreed to a meeting on Monday."}))

        result = await llm.summarize_call(
            transcript=[{"speaker": "other", "text": "Monday works for me."}],
            mission={"conversation_goal": "Schedule a meeting", "summary": ""},
        )
        assert "Monday" in result["summary"]

    async def test_summarize_call_uses_temperature_zero(self):
        from engine.modules.llm.client import AnthropicLLM

        llm = AnthropicLLM(api_key="k", model="m")
        recorded_temp = []

        async def fake_chat(system, user, temperature, max_tokens):
            recorded_temp.append(temperature)
            return json.dumps({"summary": "done"})

        llm._chat = fake_chat
        await llm.summarize_call([], {})
        assert recorded_temp[0] == pytest.approx(0.0)


class TestRepairLLMResponse:
    def test_strips_markdown_fences(self):
        from engine.modules.llm.client import repair_llm_response
        raw = "```json\n{\"steps\": []}\n```"
        result = repair_llm_response(raw)
        assert result == {"steps": []}

    def test_removes_trailing_commas(self):
        from engine.modules.llm.client import repair_llm_response
        raw = '{"steps": [1, 2,]}'
        result = repair_llm_response(raw)
        assert result["steps"] == [1, 2]

    def test_truncates_to_four_options(self):
        from engine.modules.llm.client import repair_llm_response
        options = [{"id": i, "text": f"opt{i}", "tone": "direct", "recommended": False} for i in range(1, 7)]
        raw = json.dumps({"options": options})
        result = repair_llm_response(raw)
        assert len(result["options"]) == 4

    def test_promotes_first_option_when_none_recommended(self):
        from engine.modules.llm.client import repair_llm_response
        options = [{"id": i, "text": f"opt{i}", "tone": "direct", "recommended": False} for i in range(1, 5)]
        raw = json.dumps({"options": options})
        result = repair_llm_response(raw)
        assert result["options"][0]["recommended"] is True

    def test_adds_missing_recommended_field(self):
        from engine.modules.llm.client import repair_llm_response
        options = [{"id": i, "text": f"opt{i}", "tone": "direct"} for i in range(1, 5)]
        raw = json.dumps({"options": options})
        result = repair_llm_response(raw)
        assert all("recommended" in o for o in result["options"])

    def test_raises_on_invalid_json(self):
        from engine.modules.llm.client import repair_llm_response
        import json as _json
        with pytest.raises(_json.JSONDecodeError):
            repair_llm_response("not json at all {{{")

"""Unit tests for reasoning-client JSON helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from art_style_search.reasoning_client import ReasoningClient, parse_json_response


class TestParseJsonResponse:
    def test_handles_markdown_fences(self) -> None:
        data = parse_json_response(
            """```json
{"ok": true, "count": 2}
```"""
        )
        assert data == {"ok": True, "count": 2}

    def test_extracts_json_from_preamble(self) -> None:
        data = parse_json_response('Here is the result:\n{"status": "done"}')
        assert data == {"status": "done"}


class TestCallJson:
    @pytest.mark.asyncio
    async def test_repairs_invalid_json_once(self) -> None:
        client = ReasoningClient.__new__(ReasoningClient)
        responses = iter(["not json", '{"status": "fixed"}'])

        async def fake_call(*, model, system, user, max_tokens):
            return next(responses)

        client.call = fake_call  # type: ignore[method-assign]

        result = await ReasoningClient.call_json(
            client,
            model="fake-model",
            system="system",
            user="user",
            validator=lambda data: data["status"],  # type: ignore[index]
            response_name="test_payload",
            schema_hint='{"status": "..."}',
            repair_retries=1,
        )

        assert result == "fixed"


class TestCallText:
    @pytest.mark.asyncio
    async def test_xai_uses_role_based_input_without_reasoning_effort(self) -> None:
        captured: dict[str, object] = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text="xai ok")

        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "xai"
        client._xai = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

        result = await ReasoningClient.call(
            client,
            model="grok-4.20-reasoning-latest",
            system="system prompt",
            user="user prompt",
            max_tokens=321,
        )

        assert result == "xai ok"
        assert captured == {
            "model": "grok-4.20-reasoning-latest",
            "input": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ],
            "max_output_tokens": 321,
            "store": False,
        }

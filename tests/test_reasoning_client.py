"""Unit tests for reasoning-client JSON helpers."""

from __future__ import annotations

import logging
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

    def test_extracts_first_complete_json_object_when_trailing_data_exists(self) -> None:
        data = parse_json_response('{"status": "done"}\n{"status": "duplicate"}')
        assert data == {"status": "done"}


class TestCallJson:
    @pytest.mark.asyncio
    async def test_openai_call_json_uses_structured_outputs(self) -> None:
        captured: dict[str, object] = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text='{"status":"fixed"}')

        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "openai"
        client._openai = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

        result = await ReasoningClient.call_json(
            client,
            model="gpt-5.4",
            system="system",
            user="user",
            validator=lambda data: data["status"],  # type: ignore[index]
            response_name="test_payload",
            schema_hint='{"status": "..."}',
        )

        assert result == "fixed"
        assert captured["model"] == "gpt-5.4"
        assert captured["input"] == "user"
        assert captured["instructions"] == "system"
        assert captured["text"] == {
            "format": {
                "description": "Structured response for test_payload",
                "name": "test_payload",
                "schema": {"additionalProperties": True, "type": "object"},
                "strict": True,
                "type": "json_schema",
            }
        }

    @pytest.mark.asyncio
    async def test_repairs_invalid_json_once_logs_attempt_at_info(self, caplog) -> None:
        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "anthropic"
        responses = iter(["not json", '{"status": "fixed"}'])

        async def fake_call(*, model, system, user, max_tokens):
            return next(responses)

        client.call = fake_call  # type: ignore[method-assign]

        with caplog.at_level("INFO"):
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
        assert "attempting JSON repair" in caplog.text
        assert not [record for record in caplog.records if record.levelno >= logging.WARNING]

    @pytest.mark.asyncio
    async def test_retries_validation_after_repair_with_followup_feedback(self) -> None:
        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "anthropic"
        responses = iter(
            [
                '{"status": 1}',
                '{"status": 2}',
                '{"status": "fixed"}',
            ]
        )

        async def fake_call(*, model, system, user, max_tokens):
            return next(responses)

        client.call = fake_call  # type: ignore[method-assign]

        def validator(data: object) -> str:
            value = data["status"]  # type: ignore[index]
            if not isinstance(value, str):
                raise ValueError("status must be a string")
            return value

        result = await ReasoningClient.call_json(
            client,
            model="fake-model",
            system="system",
            user="user",
            validator=validator,
            response_name="test_payload",
            schema_hint='{"status": "..."}',
            repair_retries=2,
        )

        assert result == "fixed"

    @pytest.mark.asyncio
    async def test_logs_final_failure_at_warning_by_default(self, caplog) -> None:
        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "anthropic"
        responses = iter(["not json", "still not json"])

        async def fake_call(*, model, system, user, max_tokens):
            return next(responses)

        client.call = fake_call  # type: ignore[method-assign]

        with caplog.at_level("INFO"), pytest.raises(RuntimeError, match="test_payload validation failed after repair"):
            await ReasoningClient.call_json(
                client,
                model="fake-model",
                system="system",
                user="user",
                validator=lambda data: data["status"],  # type: ignore[index]
                response_name="test_payload",
                schema_hint='{"status": "..."}',
                repair_retries=1,
            )

        assert any(record.levelno == logging.WARNING for record in caplog.records)

    @pytest.mark.asyncio
    async def test_supports_info_level_for_noncritical_final_failure(self, caplog) -> None:
        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "anthropic"
        responses = iter(["not json", "still not json"])

        async def fake_call(*, model, system, user, max_tokens):
            return next(responses)

        client.call = fake_call  # type: ignore[method-assign]

        with caplog.at_level("INFO"), pytest.raises(RuntimeError, match="test_payload validation failed after repair"):
            await ReasoningClient.call_json(
                client,
                model="fake-model",
                system="system",
                user="user",
                validator=lambda data: data["status"],  # type: ignore[index]
                response_name="test_payload",
                schema_hint='{"status": "..."}',
                repair_retries=1,
                final_failure_log_level=logging.INFO,
            )

        assert "validation failed after 1 repair attempt" in caplog.text
        assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


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

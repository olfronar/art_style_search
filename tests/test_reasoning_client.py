"""Unit tests for reasoning-client JSON helpers."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from art_style_search.reasoning_client import (
    ReasoningClient,
    parse_json_response,
)


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
        client._openai = SimpleNamespace(responses=SimpleNamespace(create=fake_create))  # type: ignore[assignment]

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

        async def fake_call(
            *, model, system, user, max_tokens, temperature=None, reasoning_effort=None, stage="unknown"
        ):
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

        async def fake_call(
            *, model, system, user, max_tokens, temperature=None, reasoning_effort=None, stage="unknown"
        ):
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

        async def fake_call(
            *, model, system, user, max_tokens, temperature=None, reasoning_effort=None, stage="unknown"
        ):
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

        async def fake_call(
            *, model, system, user, max_tokens, temperature=None, reasoning_effort=None, stage="unknown"
        ):
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
        client._xai = SimpleNamespace(responses=SimpleNamespace(create=fake_create))  # type: ignore[assignment]

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


class TestAnthropicBudgetGuard:
    @pytest.mark.asyncio
    async def test_raises_when_budget_exceeds_max_tokens(self) -> None:
        from art_style_search.reasoning_client import _ANTHROPIC_HIGH_BUDGET_TOKENS

        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "anthropic"
        client._anthropic = SimpleNamespace()  # type: ignore[assignment]

        with pytest.raises(ValueError, match="budget_tokens"):
            await ReasoningClient.call(
                client,
                model="claude-opus-4-7",
                system="system",
                user="user",
                max_tokens=_ANTHROPIC_HIGH_BUDGET_TOKENS,
                reasoning_effort="high",
            )

    @pytest.mark.asyncio
    async def test_allows_when_budget_below_max_tokens(self, monkeypatch) -> None:
        from art_style_search.reasoning_client import _ANTHROPIC_HIGH_BUDGET_TOKENS

        captured: dict[str, object] = {}

        async def fake_stream(client, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])

        monkeypatch.setattr("art_style_search.reasoning_client.stream_message", fake_stream)

        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "anthropic"
        client._anthropic = SimpleNamespace()  # type: ignore[assignment]

        result = await ReasoningClient.call(
            client,
            model="claude-opus-4-7",
            system="system",
            user="user",
            max_tokens=_ANTHROPIC_HIGH_BUDGET_TOKENS + 1000,
            reasoning_effort="high",
        )

        assert result == "ok"
        assert captured["thinking"] == {
            "type": "enabled",
            "budget_tokens": _ANTHROPIC_HIGH_BUDGET_TOKENS,
        }


class TestSilentDropWarnings:
    @pytest.mark.asyncio
    async def test_zai_warns_once_per_process(self, caplog) -> None:
        from art_style_search.reasoning_client import _silent_drop_warned

        _silent_drop_warned.discard(("zai", "reasoning_effort"))

        class FakeCompletions:
            def create(self, **_):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "zai"
        client._zai = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))  # type: ignore[assignment]

        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                await ReasoningClient.call(
                    client,
                    model="glm-5.1",
                    system="system",
                    user="user",
                    max_tokens=1000,
                    reasoning_effort="high",
                )

        drop_records = [r for r in caplog.records if "silently drops" in r.getMessage()]
        assert len(drop_records) == 1
        assert "zai" in drop_records[0].getMessage()

    @pytest.mark.asyncio
    async def test_openai_warns_once_on_temperature(self, caplog) -> None:
        from art_style_search.reasoning_client import _silent_drop_warned

        _silent_drop_warned.discard(("openai", "temperature"))

        async def fake_create(**_):
            return SimpleNamespace(output_text="ok")

        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "openai"
        client._openai = SimpleNamespace(responses=SimpleNamespace(create=fake_create))  # type: ignore[assignment]

        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                await ReasoningClient.call(
                    client,
                    model="gpt-5.4",
                    system="system",
                    user="user",
                    max_tokens=1000,
                    temperature=0.9,
                )

        drop_records = [r for r in caplog.records if "silently drops" in r.getMessage()]
        assert len(drop_records) == 1
        assert "openai" in drop_records[0].getMessage()


class TestGeminiTimeoutScaling:
    def test_floor_for_tiny_budget(self) -> None:
        from art_style_search.retry import gemini_timeout_s

        assert gemini_timeout_s(0) == 60.0
        assert gemini_timeout_s(1000) == 60.0  # 30 + 20 = 50 → clamped to 60

    def test_scales_with_budget(self) -> None:
        from art_style_search.retry import gemini_timeout_s

        assert gemini_timeout_s(8000) == 190.0  # 30 + 160
        assert gemini_timeout_s(32000) == 670.0  # 30 + 640


class TestLogApiCall:
    def test_emits_structured_line(self, caplog) -> None:
        from art_style_search.retry import log_api_call

        with caplog.at_level(logging.INFO):
            log_api_call(
                provider="anthropic",
                model="claude-opus-4-7",
                stage="brainstorm",
                duration_s=2.5,
                max_tokens=40000,
                effort="medium",
                usage={"input_tokens": 100, "output_tokens": 2000},
            )

        records = [r for r in caplog.records if r.getMessage().startswith("API_CALL ")]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "provider=anthropic" in msg
        assert "model=claude-opus-4-7" in msg
        assert "stage=brainstorm" in msg
        assert "duration_s=2.50" in msg
        assert "max_tokens=40000" in msg
        assert "effort=medium" in msg
        assert "usage=input_tokens:100,output_tokens:2000" in msg
        assert "status=ok" in msg

    def test_omits_optional_fields(self, caplog) -> None:
        from art_style_search.retry import log_api_call

        with caplog.at_level(logging.INFO):
            log_api_call(provider="gemini", model="flash", stage="generate", duration_s=1.2)

        msg = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("API_CALL "))
        assert "max_tokens=" not in msg
        assert "effort=" not in msg
        assert "thinking_level=" not in msg
        assert "usage=" not in msg


class TestTruncationDetection:
    @pytest.mark.asyncio
    async def test_anthropic_truncation_raises_and_is_not_retried(self, monkeypatch) -> None:
        from art_style_search.reasoning_client import TruncationError

        call_count = {"n": 0}

        async def fake_stream(client, **_):
            call_count["n"] += 1
            usage = SimpleNamespace(input_tokens=10, output_tokens=4000)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="partial")],
                stop_reason="max_tokens",
                usage=usage,
            )

        monkeypatch.setattr("art_style_search.reasoning_client.stream_message", fake_stream)

        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "anthropic"
        client._anthropic = SimpleNamespace()  # type: ignore[assignment]

        with pytest.raises(TruncationError) as excinfo:
            await ReasoningClient.call(
                client,
                model="claude-opus-4-7",
                system="system",
                user="user",
                max_tokens=4000,
                reasoning_effort="low",
            )

        assert excinfo.value.provider == "anthropic"
        assert excinfo.value.max_tokens == 4000
        assert excinfo.value.completion_tokens == 4000
        assert call_count["n"] == 1  # not retried

    @pytest.mark.asyncio
    async def test_openai_incomplete_raises_truncation(self) -> None:
        from art_style_search.reasoning_client import TruncationError

        async def fake_create(**_):
            return SimpleNamespace(
                output_text="partial",
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=1000,
                    total_tokens=1010,
                    output_tokens_details=None,
                ),
            )

        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "openai"
        client._openai = SimpleNamespace(responses=SimpleNamespace(create=fake_create))  # type: ignore[assignment]

        with pytest.raises(TruncationError) as excinfo:
            await ReasoningClient.call(
                client,
                model="gpt-5.4",
                system="system",
                user="user",
                max_tokens=1000,
            )

        assert excinfo.value.provider == "openai"
        assert excinfo.value.completion_tokens == 1000

    @pytest.mark.asyncio
    async def test_chat_completion_length_raises_truncation(self) -> None:
        from art_style_search.reasoning_client import TruncationError

        class FakeCompletions:
            def create(self, **_):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="partial"),
                            finish_reason="length",
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=500, total_tokens=510),
                )

        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "zai"
        client._zai = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))  # type: ignore[assignment]

        with pytest.raises(TruncationError) as excinfo:
            await ReasoningClient.call(
                client,
                model="glm-5.1",
                system="system",
                user="user",
                max_tokens=500,
            )

        assert excinfo.value.provider == "zai"
        assert excinfo.value.completion_tokens == 500

    @pytest.mark.asyncio
    async def test_retry_skips_truncation(self, caplog) -> None:
        from art_style_search.reasoning_client import TruncationError
        from art_style_search.retry import async_retry

        attempts = {"n": 0}

        async def fn():
            attempts["n"] += 1
            raise TruncationError(provider="test", stage="s", max_tokens=1000)

        with caplog.at_level(logging.WARNING), pytest.raises(TruncationError):
            await async_retry(fn, max_retries=5, base_delay=0.01, label="X")

        assert attempts["n"] == 1  # no retry
        assert any("truncated" in r.getMessage() for r in caplog.records)


class TestCallWithImages:
    @pytest.mark.asyncio
    async def test_builds_multimodal_content_and_uses_low_effort(self, monkeypatch, tmp_path) -> None:
        from PIL import Image

        image = tmp_path / "ref.png"
        Image.new("RGB", (8, 8), color="blue").save(image)

        captured: dict[str, object] = {}

        async def fake_stream(client, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="caption body")])

        monkeypatch.setattr("art_style_search.reasoning_client.stream_message", fake_stream)

        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "anthropic"
        client._anthropic = SimpleNamespace()  # type: ignore[assignment]
        client.default_reasoning_effort = "medium"

        result = await ReasoningClient.call_with_images(
            client,
            model="claude-opus-4-7",
            system="system",
            user="describe",
            image_paths=[image],
            max_tokens=4000,
            reasoning_effort="low",
            stage="caption_bootstrap",
        )

        assert result == "caption body"
        messages = captured["messages"]
        assert isinstance(messages, list)
        content = messages[0]["content"]
        assert content[0]["type"] == "image"
        # Downscaled + re-encoded as JPEG regardless of source format.
        assert content[0]["source"]["media_type"] == "image/jpeg"
        assert content[0]["source"]["type"] == "base64"
        assert content[-1] == {"type": "text", "text": "describe"}
        # low effort disables thinking; no cache_control on system block for one-shot bootstrap calls
        assert captured["thinking"] == {"type": "disabled"}
        assert "cache_control" not in captured["system"][0]  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_non_anthropic_provider_raises(self, tmp_path) -> None:
        image = tmp_path / "ref.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n\x00fake")

        client = ReasoningClient.__new__(ReasoningClient)
        client.provider = "openai"

        with pytest.raises(NotImplementedError, match="anthropic"):
            await ReasoningClient.call_with_images(
                client,
                model="gpt-5.4",
                system="s",
                user="u",
                image_paths=[image],
            )

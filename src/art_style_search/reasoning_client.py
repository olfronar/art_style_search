"""Provider-agnostic reasoning client and lightweight parsing helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import random as _rng
import re
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import anthropic
import httpcore
import httpx
from anthropic.types import Message

from art_style_search.retry import async_retry

logger = logging.getLogger(__name__)

_STREAM_MAX_RETRIES = 3
_STREAM_BASE_DELAY = 5.0
T = TypeVar("T")


def _adapt_prompts_for_provider(system: str, user: str, provider: str) -> tuple[str, str]:
    """Apply lightweight provider-specific prompt adaptations.

    Returns (adapted_system, adapted_user).
    """
    if provider == "openai":
        # GPT models sometimes under-weight system prompts.
        # Append a brief reminder of the most critical rule to the user message.
        if "NON-NEGOTIABLE" in system or "CRITICAL" in system:
            user += (
                "\n\n[Reminder: Return EXACTLY one JSON object. "
                "First section must be 'style_foundation', second must be 'subject_anchor'. "
                "No markdown fences, no commentary.]"
            )
    elif provider == "zai" and "json" in system.lower():
        # GLM benefits from explicit formatting reinforcement for structured output.
        system += (
            "\n\nImportant: Respond with valid JSON only. "
            "Do not include any text before or after the JSON object. "
            "Do not wrap in markdown code blocks."
        )
    # For "anthropic", "local", "xai" — no adaptation needed (current format works well).
    return system, user


def extract_xml_tag(text: str, tag: str) -> str:
    """Extract text content between <tag> and </tag>, stripped. Returns '' if absent."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_text(response: Message) -> str:
    """Extract text content from a response that may contain thinking blocks."""
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_payload(text: str) -> str:
    """Best-effort extraction of a top-level JSON payload from model text."""
    stripped = _strip_json_fences(text)
    if not stripped:
        return stripped

    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end != -1 and end > start:
            return stripped[start : end + 1]
    return stripped


def parse_json_response(text: str) -> object:
    """Parse JSON from a model response, tolerating markdown fences or preamble."""
    stripped = _strip_json_fences(text)
    decoder = json.JSONDecoder()

    for opener in ("{", "["):
        start = stripped.find(opener)
        if start == -1:
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[start:])
            return obj
        except json.JSONDecodeError:
            continue

    payload = _extract_json_payload(text)
    return json.loads(payload)


async def stream_message(client: anthropic.AsyncAnthropic, **kwargs: object) -> Message:
    """Call messages.create with streaming and return the final Message."""
    last_exc: Exception | None = None
    for attempt in range(_STREAM_MAX_RETRIES):
        try:
            async with client.messages.stream(**kwargs) as stream:
                return await stream.get_final_message()
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            last_exc = exc
            delay = _STREAM_BASE_DELAY * (2**attempt) * (0.5 + _rng.random())
            logger.warning(
                "Anthropic stream attempt %d/%d failed: %s — retrying in %.0fs",
                attempt + 1,
                _STREAM_MAX_RETRIES,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
        except (
            httpx.RemoteProtocolError,
            httpx.ReadError,
            httpcore.RemoteProtocolError,
            httpcore.ReadError,
            ConnectionResetError,
            BrokenPipeError,
        ) as exc:
            last_exc = exc
            delay = _STREAM_BASE_DELAY * (2**attempt) * (0.5 + _rng.random())
            logger.warning(
                "Anthropic stream attempt %d/%d failed: %s: %s — retrying in %.0fs",
                attempt + 1,
                _STREAM_MAX_RETRIES,
                type(exc).__name__,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
        except Exception as exc:
            exc_str = str(exc).lower()
            is_transient = "incomplete chunked read" in exc_str or "peer closed connection" in exc_str
            if is_transient:
                last_exc = exc
                delay = _STREAM_BASE_DELAY * (2**attempt) * (0.5 + _rng.random())
                logger.warning(
                    "Anthropic stream attempt %d/%d (string-match fallback): %s: %s — retrying in %.0fs",
                    attempt + 1,
                    _STREAM_MAX_RETRIES,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                raise

    msg = f"Anthropic stream failed after {_STREAM_MAX_RETRIES} retries"
    raise RuntimeError(msg) from last_exc


class ReasoningClient:
    """Wraps Anthropic, Z.AI, OpenAI, xAI, or a local OpenAI-compatible server behind a unified async interface."""

    def __init__(
        self,
        provider: str,
        *,
        anthropic_api_key: str = "",
        zai_api_key: str = "",
        openai_api_key: str = "",
        xai_api_key: str = "",
        base_url: str = "",
        debug_dir: Path | str | None = None,
    ) -> None:
        self.provider = provider
        self._debug_dir = Path(debug_dir) if debug_dir else None
        if provider == "anthropic":
            self._anthropic = anthropic.AsyncAnthropic(
                api_key=anthropic_api_key,
                timeout=anthropic.Timeout(600.0, connect=30.0),
            )
        elif provider == "zai":
            from zai import ZaiClient

            self._zai = ZaiClient(
                api_key=zai_api_key,
                timeout=httpx.Timeout(300.0, connect=15.0),
            )
        elif provider == "openai":
            from openai import AsyncOpenAI

            self._openai = AsyncOpenAI(
                api_key=openai_api_key,
                timeout=httpx.Timeout(600.0, connect=30.0),
            )
        elif provider == "xai":
            from openai import AsyncOpenAI

            self._xai = AsyncOpenAI(
                api_key=xai_api_key,
                base_url="https://api.x.ai/v1",
                timeout=httpx.Timeout(3600.0, connect=30.0),
            )
        elif provider == "local":
            from openai import AsyncOpenAI

            self._local = AsyncOpenAI(
                api_key="not-needed",
                base_url=base_url,
                timeout=httpx.Timeout(600.0, connect=30.0),
            )
        else:
            msg = f"Unknown reasoning provider: {provider}"
            raise ValueError(msg)

    async def call(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 16000,
    ) -> str:
        """Send a reasoning request and return the text response."""
        system, user = _adapt_prompts_for_provider(system, user, self.provider)
        if self.provider == "anthropic":
            return await self._call_anthropic(model=model, system=system, user=user, max_tokens=max_tokens)
        if self.provider == "openai":
            return await self._call_openai(model=model, system=system, user=user, max_tokens=max_tokens)
        if self.provider == "xai":
            return await self._call_xai(model=model, system=system, user=user, max_tokens=max_tokens)
        if self.provider == "local":
            return await self._call_local(model=model, system=system, user=user, max_tokens=max_tokens)
        return await self._call_zai(model=model, system=system, user=user, max_tokens=max_tokens)

    async def call_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        validator: Callable[[object], T],
        response_name: str,
        schema_hint: str = "",
        response_schema: dict[str, object] | None = None,
        max_tokens: int = 16000,
        repair_retries: int = 1,
        final_failure_log_level: int = logging.WARNING,
    ) -> T:
        """Send a reasoning request, parse JSON, validate it, and optionally repair it."""

        system, user = _adapt_prompts_for_provider(system, user, self.provider)
        current_text = await self._call_json_transport(
            model=model,
            system=system,
            user=user,
            response_name=response_name,
            response_schema=response_schema,
            max_tokens=max_tokens,
        )
        try:
            return validator(parse_json_response(current_text))
        except Exception as exc:
            self._write_debug_artifact(response_name, "raw", current_text)
            self._write_debug_artifact(response_name, "validation_error", str(exc))
            if repair_retries <= 0:
                msg = f"{response_name} validation failed"
                logger.log(final_failure_log_level, "%s validation failed: %s", response_name, exc)
                raise RuntimeError(msg) from exc

            current_exc = exc

        repair_system = (
            "You repair malformed model outputs into valid JSON.\n"
            "Return a single JSON object only. No markdown fences, no commentary."
        )
        hint_block = f"\nExpected schema:\n{schema_hint}\n" if schema_hint else ""
        for attempt in range(repair_retries):
            logger.info(
                "%s validation failed: %s — attempting JSON repair (%d/%d)",
                response_name,
                current_exc,
                attempt + 1,
                repair_retries,
            )
            repair_user = (
                f"The previous response for '{response_name}' was invalid.\n"
                f"Validation error: {current_exc}\n"
                f"{hint_block}"
                "Return a corrected JSON response for the same request.\n\n"
                "Invalid response:\n"
                f"{current_text}\n"
            )
            current_text = await self._call_json_transport(
                model=model,
                system=repair_system,
                user=repair_user,
                response_name=f"{response_name}_repair",
                response_schema=response_schema,
                max_tokens=max_tokens,
            )
            self._write_debug_artifact(response_name, f"repair_{attempt + 1}", current_text)
            try:
                return validator(parse_json_response(current_text))
            except Exception as repair_exc:
                self._write_debug_artifact(response_name, f"repair_{attempt + 1}_error", str(repair_exc))
                current_exc = repair_exc

        msg = f"{response_name} validation failed after repair"
        logger.log(
            final_failure_log_level,
            "%s validation failed after %d repair attempt(s): %s",
            response_name,
            repair_retries,
            current_exc,
        )
        raise RuntimeError(msg) from current_exc

    async def _call_json_transport(
        self,
        *,
        model: str,
        system: str,
        user: str,
        response_name: str,
        response_schema: dict[str, object] | None,
        max_tokens: int,
    ) -> str:
        if self.provider == "openai":
            return await self._call_openai_json(
                model=model,
                system=system,
                user=user,
                response_name=response_name,
                response_schema=response_schema,
                max_tokens=max_tokens,
            )
        return await self.call(model=model, system=system, user=user, max_tokens=max_tokens)

    async def _call_anthropic(self, *, model: str, system: str, user: str, max_tokens: int) -> str:
        response = await stream_message(
            self._anthropic,
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return extract_text(response)

    async def _call_zai(self, *, model: str, system: str, user: str, max_tokens: int) -> str:
        def _sync_call() -> str:
            response = self._zai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        async def _call() -> str:
            return await asyncio.to_thread(_sync_call)

        return await async_retry(_call, label="Z.AI call", base_delay=_STREAM_BASE_DELAY)

    async def _call_openai(self, *, model: str, system: str, user: str, max_tokens: int) -> str:
        async def _call() -> str:
            response = await self._openai.responses.create(
                model=model,
                instructions=system,
                input=user,
                reasoning={"effort": "medium"},
                max_output_tokens=max_tokens,
            )
            return response.output_text

        return await async_retry(_call, label="OpenAI call", base_delay=_STREAM_BASE_DELAY)

    async def _call_openai_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        response_name: str,
        response_schema: dict[str, object] | None,
        max_tokens: int,
    ) -> str:
        format_name = re.sub(r"[^A-Za-z0-9_-]+", "_", response_name)[:64] or "response"
        json_schema = response_schema or {"type": "object", "additionalProperties": True}

        async def _call() -> str:
            response = await self._openai.responses.create(
                model=model,
                instructions=system,
                input=user,
                reasoning={"effort": "medium"},
                max_output_tokens=max_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": format_name,
                        "description": f"Structured response for {format_name}",
                        "strict": True,
                        "schema": json_schema,
                    }
                },
            )
            return response.output_text

        return await async_retry(_call, label="OpenAI JSON call", base_delay=_STREAM_BASE_DELAY)

    async def _call_xai(self, *, model: str, system: str, user: str, max_tokens: int) -> str:
        async def _call() -> str:
            response = await self._xai.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_output_tokens=max_tokens,
                store=False,
            )
            return response.output_text

        return await async_retry(_call, label="xAI call", base_delay=_STREAM_BASE_DELAY)

    async def _call_local(self, *, model: str, system: str, user: str, max_tokens: int) -> str:
        async def _call() -> str:
            response = await self._local.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        return await async_retry(_call, label="Local model call", base_delay=2.0)

    def _write_debug_artifact(self, response_name: str, suffix: str, content: str) -> None:
        debug_dir = getattr(self, "_debug_dir", None)
        if debug_dir is None:
            return
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", response_name)
        path = debug_dir / f"{safe_name}_{suffix}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

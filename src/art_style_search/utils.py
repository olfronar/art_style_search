"""Shared utilities for Anthropic, Z.AI, and Gemini API interactions."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import anthropic
from anthropic.types import Message
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

_STREAM_MAX_RETRIES = 3
_STREAM_BASE_DELAY = 5.0

MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def image_to_gemini_part(path: Path) -> genai_types.Part:
    """Read an image file and return a Gemini Part with correct MIME type."""
    mime_type = MIME_MAP.get(path.suffix.lower(), "image/png")
    return genai_types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def extract_text(response: Message) -> str:
    """Extract text content from a response that may contain thinking blocks."""
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


async def stream_message(client: anthropic.AsyncAnthropic, **kwargs: object) -> Message:
    """Call messages.create with streaming and return the final Message.

    Retries on transient connection errors (dropped connections, incomplete reads).
    """
    import asyncio

    last_exc: Exception | None = None
    for attempt in range(_STREAM_MAX_RETRIES):
        try:
            async with client.messages.stream(**kwargs) as stream:
                return await stream.get_final_message()
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            last_exc = exc
            delay = _STREAM_BASE_DELAY * (2**attempt)
            logger.warning(
                "Claude stream attempt %d/%d failed: %s — retrying in %.0fs",
                attempt + 1,
                _STREAM_MAX_RETRIES,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
        except Exception as exc:
            exc_name = type(exc).__name__.lower()
            exc_str = str(exc).lower()
            is_transient = (
                "incomplete chunked read" in exc_str
                or "peer closed connection" in exc_str
                or "readerror" in exc_name
                or "readtimeout" in exc_name
                or "remotedisconnected" in exc_name
                or "connectionerror" in exc_name
            )
            if is_transient:
                last_exc = exc
                delay = _STREAM_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Claude stream attempt %d/%d failed: %s: %s — retrying in %.0fs",
                    attempt + 1,
                    _STREAM_MAX_RETRIES,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                raise

    msg = f"Claude stream failed after {_STREAM_MAX_RETRIES} retries"
    raise RuntimeError(msg) from last_exc


# ---------------------------------------------------------------------------
# Provider-agnostic reasoning client
# ---------------------------------------------------------------------------


class ReasoningClient:
    """Wraps either Anthropic (Claude) or Z.AI (GLM) behind a unified async interface."""

    def __init__(self, provider: str, *, anthropic_api_key: str = "", zai_api_key: str = "") -> None:
        self.provider = provider
        if provider == "anthropic":
            self._anthropic = anthropic.AsyncAnthropic(
                api_key=anthropic_api_key,
                timeout=anthropic.Timeout(600.0, connect=30.0),
            )
        elif provider == "zai":
            from zai import ZaiClient

            self._zai = ZaiClient(api_key=zai_api_key)
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
        if self.provider == "anthropic":
            return await self._call_anthropic(model=model, system=system, user=user, max_tokens=max_tokens)
        return await self._call_zai(model=model, system=system, user=user, max_tokens=max_tokens)

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
        """Call Z.AI synchronous SDK via asyncio.to_thread with retry."""

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

        last_exc: Exception | None = None
        for attempt in range(_STREAM_MAX_RETRIES):
            try:
                return await asyncio.to_thread(_sync_call)
            except Exception as exc:
                last_exc = exc
                delay = _STREAM_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Z.AI call attempt %d/%d failed: %s: %s — retrying in %.0fs",
                    attempt + 1,
                    _STREAM_MAX_RETRIES,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        msg = f"Z.AI call failed after {_STREAM_MAX_RETRIES} retries"
        raise RuntimeError(msg) from last_exc

"""Shared utilities for Anthropic and Gemini API interactions."""

from __future__ import annotations

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
            exc_str = str(exc).lower()
            is_transient = (
                "incomplete chunked read" in exc_str
                or "peer closed connection" in exc_str
                or "readtimeout" in type(exc).__name__.lower()
                or "read timed out" in exc_str
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

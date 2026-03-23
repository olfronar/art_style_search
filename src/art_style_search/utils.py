"""Shared utilities for Anthropic and Gemini API interactions."""

from __future__ import annotations

from pathlib import Path

import anthropic
from anthropic.types import Message
from google.genai import types as genai_types

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
    """Call messages.create with streaming and return the final Message."""
    async with client.messages.stream(**kwargs) as stream:
        return await stream.get_final_message()

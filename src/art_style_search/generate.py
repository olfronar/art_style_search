"""Image generation via Gemini Flash with semaphore throttling and retry logic."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY = 2.0


async def _generate_single(
    prompt: str,
    *,
    index: int,
    aspect_ratio: str,
    output_path: Path,
    client: genai.Client,
    model: str,
    semaphore: asyncio.Semaphore,
) -> Path:
    """Generate a single image with semaphore gating and exponential backoff."""
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        async with semaphore:
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        thinking_config=genai_types.ThinkingConfig(thinking_level="MINIMAL"),
                        image_config=genai_types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                            image_size="1K",
                        ),
                    ),
                )

                if not response.candidates or not response.candidates[0].content.parts:
                    msg = f"Image {index}: empty response from model"
                    raise RuntimeError(msg)

                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        output_path.write_bytes(part.inline_data.data)
                        return output_path
                    if hasattr(part, "image") and part.image is not None:
                        output_path.write_bytes(part.image.image_bytes)
                        return output_path

                part_types = [type(p).__name__ for p in response.candidates[0].content.parts]
                text_parts = [p.text for p in response.candidates[0].content.parts if hasattr(p, "text") and p.text]
                text_summary = "; ".join(t[:200] for t in text_parts) if text_parts else "none"
                logger.warning(
                    "Image %d: response parts=%s, text=%s",
                    index,
                    part_types,
                    text_summary,
                )
                msg = f"Image {index}: no image data found in response parts"
                raise RuntimeError(msg)

            except genai.errors.ClientError as exc:
                last_exc = exc
                delay = _BASE_DELAY * (2**attempt)
                logger.warning(
                    "Image %d: ClientError on attempt %d/%d, retrying in %.1fs: %s",
                    index,
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
            except Exception as exc:
                # Catch rate-limit or transient errors
                last_exc = exc
                delay = _BASE_DELAY * (2**attempt)
                logger.warning(
                    "Image %d: error on attempt %d/%d, retrying in %.1fs: %s",
                    index,
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

    msg = f"Image {index}: all {_MAX_RETRIES} retries exhausted"
    raise RuntimeError(msg) from last_exc

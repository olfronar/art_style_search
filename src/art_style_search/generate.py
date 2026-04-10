"""Image generation via Gemini Flash with semaphore throttling and retry logic."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from google import genai
from google.genai import types as genai_types

from art_style_search.utils import async_retry, gemini_circuit_breaker

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 180  # seconds — per-request timeout to release semaphore on hang


async def generate_single(
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
    # Disk cache: skip API call if image already exists (e.g. crash+resume)
    try:
        if output_path.stat().st_size > 0:
            logger.debug("Image %d: cached at %s", index, output_path)
            return output_path
    except OSError:
        pass

    async def _call() -> Path:
        async with semaphore:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
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
                ),
                timeout=_REQUEST_TIMEOUT,
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

    return await async_retry(_call, label=f"Image {index}", base_delay=4.0, circuit_breaker=gemini_circuit_breaker)

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
                        response_modalities=["TEXT", "IMAGE"],
                        image_generation_config=genai_types.ImageGenerationConfig(
                            aspect_ratio=aspect_ratio,
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

                msg = f"Image {index}: no inline_data found in response parts"
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
            except RuntimeError:
                raise
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


async def generate_images(
    prompt: str,
    *,
    num_images: int,
    aspect_ratio: str,
    output_dir: Path,
    iteration: int,
    branch_id: int,
    client: genai.Client,
    model: str,
    semaphore: asyncio.Semaphore,
) -> list[Path]:
    """Generate num_images images, save to disk, return paths."""
    iter_dir = output_dir / f"iter_{iteration:03d}" / f"branch_{branch_id}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        _generate_single(
            prompt,
            index=i,
            aspect_ratio=aspect_ratio,
            output_path=iter_dir / f"{i:02d}.png",
            client=client,
            model=model,
            semaphore=semaphore,
        )
        for i in range(num_images)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    paths: list[Path] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning("Image %d generation failed: %s", i, result)
        else:
            paths.append(result)

    return paths

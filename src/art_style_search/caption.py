"""Caption reference images via Gemini Pro with disk caching."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from google import genai

from art_style_search.types import Caption
from art_style_search.utils import image_to_gemini_part

logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "You are an expert art analyst. Describe this image in comprehensive detail for someone who cannot see it. "
    "Structure your response with these clearly labeled sections:\n\n"
    "**Colors**: Dominant palette (name specific colors), color relationships, saturation, temperature, gradients.\n"
    "**Composition**: Layout, focal points, balance, use of space, perspective, framing.\n"
    "**Technique**: Medium (oil, watercolor, digital, etc.), brushwork or rendering style, line quality, "
    "level of detail, abstraction vs realism.\n"
    "**Textures**: Surface qualities, patterns, tactile impressions.\n"
    "**Mood & Atmosphere**: Emotional tone, lighting quality, sense of time or place.\n"
    "**Subjects**: What is depicted, their poses, expressions, relationships.\n\n"
    "Be precise and specific. Use art terminology where appropriate. "
    "Aim for 200-400 words. Do not speculate about the artist's intent; describe only what is visible."
)


async def _caption_single(
    image_path: Path,
    *,
    prompt: str,
    model: str,
    client: genai.Client,
    cache_dir: Path | None,
    semaphore: asyncio.Semaphore,
    cache_key: str = "",
) -> Caption:
    """Caption a single image, optionally using disk cache.

    When *cache_dir* is provided and *cache_key* matches the stored key,
    the cached result is returned.  The cache_key should change whenever
    the prompt changes (e.g. hash or iteration number) to invalidate stale
    entries.
    """
    if cache_dir is not None:
        cache_file = cache_dir / f"{image_path.stem}.json"
        current_mtime = image_path.stat().st_mtime

        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if cached.get("mtime") == current_mtime and cached.get("cache_key", "") == cache_key:
                    logger.debug("Cache hit for %s", image_path.name)
                    return Caption(image_path=Path(cached["image_path"]), text=cached["text"])
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt cache file %s, will re-caption", cache_file)

    # Cache miss — call Gemini with retry on transient errors
    logger.info("Captioning %s via %s", image_path.name, model)

    import asyncio as _asyncio

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            async with semaphore:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=[
                        image_to_gemini_part(image_path),
                        prompt,
                    ],
                )
            break
        except Exception as exc:
            last_exc = exc
            delay = 3.0 * (2**attempt)
            logger.warning(
                "Caption %s attempt %d/3 failed: %s: %s — retrying in %.0fs",
                image_path.name,
                attempt + 1,
                type(exc).__name__,
                exc,
                delay,
            )
            await _asyncio.sleep(delay)
    else:
        msg = f"Captioning {image_path.name} failed after 3 retries"
        raise RuntimeError(msg) from last_exc

    caption_text = response.text

    # Validate caption quality — empty or very short captions waste downstream cycles
    min_caption_length = 50
    if not caption_text or len(caption_text.strip()) < min_caption_length:
        msg = (
            f"Captioning {image_path.name} produced empty or too-short caption "
            f"({len(caption_text.strip()) if caption_text else 0} chars, min {min_caption_length})"
        )
        raise RuntimeError(msg)

    # Write cache
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "image_path": str(image_path),
            "text": caption_text,
            "mtime": image_path.stat().st_mtime,
            "cache_key": cache_key,
        }
        cache_file = cache_dir / f"{image_path.stem}.json"
        cache_file.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Cached caption for %s", image_path.name)

    return Caption(image_path=image_path, text=caption_text)


async def caption_references(
    reference_paths: list[Path],
    *,
    model: str,
    client: genai.Client,
    cache_dir: Path,
    semaphore: asyncio.Semaphore,
    prompt: str | None = None,
    cache_key: str = "",
) -> list[Caption]:
    """Caption all reference images concurrently with disk caching.

    When *prompt* is None, uses the default CAPTION_PROMPT.
    The *cache_key* invalidates cached entries when the prompt changes.
    """
    effective_prompt = prompt or CAPTION_PROMPT
    tasks = [
        _caption_single(
            path,
            prompt=effective_prompt,
            model=model,
            client=client,
            cache_dir=cache_dir,
            semaphore=semaphore,
            cache_key=cache_key,
        )
        for path in reference_paths
    ]
    return list(await asyncio.gather(*tasks))

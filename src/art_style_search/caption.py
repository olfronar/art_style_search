"""Caption reference images via Gemini Pro with disk caching."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from google import genai  # type: ignore[attr-defined]
from google.genai import types as genai_types  # type: ignore[attr-defined]

from art_style_search.types import Caption
from art_style_search.utils import async_retry, caption_circuit_breaker, image_to_gemini_part

logger = logging.getLogger(__name__)

CAPTION_SYSTEM = (
    "You are an expert art analyst. "
    "Produce precise, evidence-grounded descriptions in the exact labeled-section format requested. "
    "Do not add commentary outside the requested sections, and do not speculate beyond visible evidence."
)

CAPTION_PROMPT = (
    "Describe this image in comprehensive detail for someone who cannot see it. "
    "These descriptions will be used to understand and reproduce the art style.\n\n"
    "## Output format\n"
    "Use these labeled sections in this exact order. Each section: 2-4 sentences with specific details.\n\n"
    "[Art Style]: Shared style DNA visible in the image — recurring rendering rules, medium cues, "
    "and reusable technique language that would help recreate another image in the same style.\n"
    "[Subject] (MOST IMPORTANT): What is depicted — identity, species, poses, expressions, "
    "clothing or equipment, relationships between figures, props, and distinguishing features.\n"
    "[Color Palette]: Dominant palette (name specific colors like 'burnt sienna', not just 'brown'), "
    "color relationships, saturation, temperature, gradients.\n"
    "[Technique]: Medium (oil, watercolor, digital, etc.), brushwork or rendering style, line quality, "
    "level of detail, abstraction vs realism.\n"
    "[Composition]: Layout, focal points, balance, use of space, perspective, framing.\n"
    "[Lighting & Atmosphere]: Light direction, shadow treatment, emotional tone, sense of time or place.\n"
    "[Textures]: Surface qualities, patterns, tactile impressions.\n\n"
    "## Constraints\n"
    "- Be precise. Use art terminology and specific color names.\n"
    "- Target 400-600 words total.\n"
    "- Do not speculate about the artist's intent; describe only what is visible.\n"
    "- [Subject] must be the most detailed section (aim for 80-140 words).\n"
    "- Keep the labels exactly as written above."
)


async def caption_single(
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

        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("mtime") == current_mtime and cached.get("cache_key", "") == cache_key:
                logger.debug("Cache hit for %s", image_path.name)
                return Caption(image_path=Path(cached["image_path"]), text=cached["text"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

    # Cache miss — call Gemini with retry on transient errors
    logger.info("Captioning %s via %s", image_path.name, model)

    async def _call() -> str:
        async with semaphore:
            resp = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model,
                    contents=[
                        image_to_gemini_part(image_path),
                        prompt,
                    ],
                    config=genai_types.GenerateContentConfig(system_instruction=CAPTION_SYSTEM),
                ),
                timeout=90,
            )
        return resp.text

    caption_text: str = await async_retry(
        _call, label=f"Caption {image_path.name}", circuit_breaker=caption_circuit_breaker
    )

    # Validate caption quality — empty or very short captions waste downstream cycles
    min_caption_length = 900 if prompt == CAPTION_PROMPT else 150
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
            "mtime": current_mtime,
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
        caption_single(
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
    results = await asyncio.gather(*tasks, return_exceptions=True)
    captions: list[Caption] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning("Caption %d (%s) failed: %s", i, reference_paths[i].name, result)
        else:
            captions.append(result)
    return captions

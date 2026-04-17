"""Caption reference images via Gemini Pro with disk caching."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from google import genai  # type: ignore[attr-defined]
from google.genai import types as genai_types  # type: ignore[attr-defined]

from art_style_search.caption_sections import parse_labeled_sections
from art_style_search.types import Caption
from art_style_search.utils import async_retry, caption_circuit_breaker, image_to_gemini_part

logger = logging.getLogger(__name__)

_ANCHOR_SECTION_MIN_WORDS: dict[str, int] = {
    "Art Style": 100,
    "Subject": 80,
}

CAPTION_SYSTEM = (
    "You are an expert art analyst producing captions that function as both faithful descriptions and "
    "ready-to-use text-to-image prompts. Your captions will be fed directly to an image generator, so every "
    "detail you write must be specific enough to reproduce the original image visually.\n\n"
    "Quality standards:\n"
    "- Lead with concrete subject identity and scene-defining information before softer stylistic interpretation.\n"
    "- Prefer direct positive phrasing and generation-ready wording over hedged narration.\n"
    "- When dense specifics help, use compact comma-delimited visual tokens; one observation per sentence; "
    "tie-break shorter when the signal is the same. Prefer signal over filler — epistemic hedges are signal.\n"
    "- Use precise color names (e.g. 'burnt sienna', 'cerulean blue'), never vague terms like 'warm colors' or 'dark tones'. "
    "If you cannot name a specific color, describe it by hue, saturation, and value.\n"
    "- Quantify when possible: 'occupies roughly the left third' not 'on the left side'; '3-4 visible characters' not 'several characters'.\n"
    "- Never speculate about artist intent or historical context. Describe only what is visible.\n\n"
    "Medium identification — describe the medium in plain, observable vocabulary grounded in what you actually see. "
    "Say what the surface is: hand-painted 2D, digital vector, stylized CGI render, photoreal CGI, mixed/2.5D, "
    "or whatever fits. Pick technique words (brushstroke / paper tooth / uniform fill / bezier curve / bevel / "
    "subdivision / ambient occlusion / rim light / PBR roughness / etc.) that match the observed surface — don't "
    "mix vocabulary from incompatible media in the same image (no PBR roughness on a flat vector fill; no "
    "brushwork on a mathematically smooth CGI surface).\n\n"
    "Observations-vs-rules rule (the single most important rule):\n"
    "- [Art Style] holds RULES that apply to every image in this style. Per-image observations — specific body parts, "
    "named objects, proper nouns, actual colors, pose details — belong in [Subject], [Color Palette], [Composition], "
    "or [Lighting & Atmosphere], NEVER in [Art Style]. A sentence inside [Art Style] is well-formed only if it would "
    "still be true of a DIFFERENT image in the same style. When an upstream rule is named in the meta-prompt, "
    "cite it by name in downstream sections; never restate its content.\n\n"
    "[Art Style] 5-slot skeleton (describe HOW the style is rendered; never NAME the genre). "
    "Phrases like '3D CGI of X', 'cel-shaded anime', '{Artist}-style', 'watercolor illustration' are forbidden. "
    "Cover each slot in 2-4 sentences, tie-break shorter. Paraphrase slot labels in your own voice; do not echo headings.\n"
    "  1. How to Draw: medium identification (plain observable vocabulary — e.g. 'hand-painted 2D with soft "
    "gradient shading', 'stylized CGI render with beveled volumes and matte-plastic surfaces') + 3-5 cues; "
    "construction order (silhouette → forms → details); line policy (none / thin uniform / variable ink / painterly / crease-only).\n"
    "  2. Shading & Light: shading-layer stack in order (base → AO → midtones → rim → specular; omit absent); "
    "edge softness (hard / feathered / graduated); key-fill-rim direction and temperature. No specific body parts or objects.\n"
    "  3. Color Principle: palette family in generic terms ('saturated complementary blues'); value structure; "
    "saturation policy. No actual colors named; image-specific hues go in [Color Palette].\n"
    "  4. Surface & Texture: grain type (paper tooth / film grain / stippling / zero grain); material vocabulary "
    "(matte plastic / fondant / PBR roughness / impasto / watercolor wash) — class-appropriate only. No specific objects.\n"
    "  5. Style Invariants: 3-5 MUST/NEVER rules that every image in this style obeys — e.g. 'MUST: every character "
    "has exactly one exaggerated feature'; 'NEVER: more than three saturated hues in one frame'. Generative rules: "
    "they describe what a NEW image in this style would have to do, not what THIS image does.\n"
    "Character proportions (heads-tall, archetype, silhouette primitives) belong in the [Subject] Proportions sub-block, NOT here. "
    "[Art Style] length target: 400-800 words across the 5 slots.\n\n"
    "Output-format discipline:\n"
    "- Write section labels as PLAIN TEXT: `[Art Style]`, `[Subject]`. Never wrap in markdown bolding, backticks, or angle brackets.\n"
    "- Emit labeled blocks in the exact order the meta-prompt's ## Caption Sections block specifies. "
    "Target the word count in ## Caption Length Target. Let [Subject] carry the per-image weight; "
    "keep [Art Style] at 400-800 words of rule content only.\n\n"
    "Forbidden terms (never use — no exceptions):\n"
    "  cartoon, cartoonish, stylised, stylized (as bare adjective), beautiful, epic, whimsical, charming, cinematic."
)

CAPTION_PROMPT = (
    "Describe this image in comprehensive detail for someone who cannot see it. "
    "These descriptions will be used to understand and reproduce the art style.\n\n"
    "## Output format\n"
    "Use these labeled sections in this exact order. Keep the labels exactly as written.\n\n"
    "[Art Style]: Shared style DNA — RULES that apply to every image in this style, never per-image observations. "
    "Describe HOW the style is rendered, never NAME the genre. Cover 5 slots in 2-4 sentences each in your own voice: "
    "(1) How to Draw (medium identification in plain observable vocabulary + construction + line policy), "
    "(2) Shading & Light (shading-layer stack + edge softness + key-fill-rim direction), "
    "(3) Color Principle (palette family in generic terms + value structure + saturation policy), "
    "(4) Surface & Texture (grain + class-appropriate material vocabulary), "
    "(5) Style Invariants (3-5 MUST/NEVER rules that every image in this style obeys). "
    "No specific body parts, named objects, or actual colors here — those belong in the sections below. "
    "No style labels like '3D CGI of X', 'cel-shaded anime', or '{Artist}-style'.\n"
    "[Subject] (MOST IMPORTANT): What is depicted — identity, species, poses, expressions, "
    "clothing or equipment, relationships between figures, props, distinguishing features, character proportions "
    "(heads-tall + archetype).\n"
    "[Color Palette]: Dominant palette (name specific colors like 'burnt sienna', not just 'brown'), "
    "color relationships, saturation, temperature, gradients.\n"
    "[Composition]: Layout, focal points, balance, use of space, perspective, framing.\n"
    "[Lighting & Atmosphere]: Light direction, shadow treatment, emotional tone, sense of time or place.\n\n"
    "## Constraints\n"
    "- Be precise. Use art terminology and specific color names where they belong.\n"
    "- Target 1500-4000 words total.\n"
    "- Do not speculate about the artist's intent; describe only what is visible.\n"
    "- [Subject] is the longest section, 800-2000 words when the image supports it.\n"
    "- [Art Style] holds generic rules only, 400-800 words.\n"
    "- Ancillary sections usually land in the 150-400 word range.\n"
    "- Keep the labels exactly as written above."
)

_CAPTION_TARGET_RE = re.compile(r"target length:\s*approximately\s*(\d+)\s*words", re.IGNORECASE)
_CAPTION_TARGET_RANGE_RE = re.compile(r"target\s+(\d+)\s*-\s*(\d+)\s*words", re.IGNORECASE)
_AVG_WORD_CHARS = 4
_FALLBACK_MIN_CAPTION_CHARS = 600
_CAPTIONER_MAX_OUTPUT_TOKENS = 32000


def _caption_length_target_from_prompt(prompt: str) -> int:
    match = _CAPTION_TARGET_RE.search(prompt)
    if match:
        return int(match.group(1))
    range_match = _CAPTION_TARGET_RANGE_RE.search(prompt)
    if range_match:
        return int(range_match.group(1))
    return 0


def _minimum_caption_chars(prompt: str) -> int:
    target = _caption_length_target_from_prompt(prompt)
    if target > 0:
        return max(_FALLBACK_MIN_CAPTION_CHARS, int(target * 0.5 * _AVG_WORD_CHARS))
    return _FALLBACK_MIN_CAPTION_CHARS


async def caption_single(
    image_path: Path,
    *,
    prompt: str,
    model: str,
    client: genai.Client,
    cache_dir: Path | None,
    semaphore: asyncio.Semaphore,
    cache_key: str = "",
    thinking_level: str = "MINIMAL",
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

    # Gemini 3.1 Pro rejects thinking_level="MINIMAL" (only LOW/MEDIUM/HIGH are valid).
    # Leave thinking_config unset for MINIMAL — restores the pre-flag behavior and
    # lets the model pick its own default thinking depth.
    config_kwargs: dict[str, object] = {
        "system_instruction": CAPTION_SYSTEM,
        "max_output_tokens": _CAPTIONER_MAX_OUTPUT_TOKENS,
    }
    if thinking_level != "MINIMAL":
        config_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_level=thinking_level)

    async def _call() -> str:
        async with semaphore:
            resp = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model,
                    contents=[
                        image_to_gemini_part(image_path),
                        prompt,
                    ],
                    config=genai_types.GenerateContentConfig(**config_kwargs),
                ),
                timeout=90,
            )
        return resp.text

    caption_text: str = await async_retry(
        _call, label=f"Caption {image_path.name}", circuit_breaker=caption_circuit_breaker
    )

    # Validate caption quality — empty or very short captions waste downstream cycles
    min_caption_length = _minimum_caption_chars(prompt)
    if not caption_text or len(caption_text.strip()) < min_caption_length:
        msg = (
            f"Captioning {image_path.name} produced empty or too-short caption "
            f"({len(caption_text.strip()) if caption_text else 0} chars, min {min_caption_length})"
        )
        raise RuntimeError(msg)

    # Validate per-section anchor minima — catches catastrophic section collapses (e.g. 9-word [Art Style])
    # that pass the total-length check and poison downstream scoring.
    parsed_sections = parse_labeled_sections(caption_text)
    if parsed_sections:
        section_violations: list[str] = []
        for section_name, min_words in _ANCHOR_SECTION_MIN_WORDS.items():
            if section_name not in parsed_sections:
                continue
            actual_words = len(parsed_sections[section_name].split())
            if actual_words < min_words:
                section_violations.append(f"[{section_name}]={actual_words}w (min {min_words})")
        if section_violations:
            msg = f"Captioning {image_path.name} produced catastrophically short anchor sections: " + "; ".join(
                section_violations
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
    thinking_level: str = "MINIMAL",
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
            thinking_level=thinking_level,
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

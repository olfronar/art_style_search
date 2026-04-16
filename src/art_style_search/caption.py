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
    "- When dense specifics help, use compact comma-delimited visual tokens.\n"
    "- Use precise color names (e.g. 'burnt sienna', 'cerulean blue'), never vague terms like 'warm colors' or 'dark tones'. "
    "If you cannot name a specific color, describe it by hue, saturation, and value (e.g. 'a muted orange-brown, medium saturation, mid-value').\n"
    "- Quantify when possible: 'occupies roughly the left third' not 'on the left side'; '3-4 visible characters' not 'several characters'.\n"
    "- Describe spatial relationships with concrete positions, not just 'near' or 'behind'.\n"
    "- Never speculate about artist intent or historical context. Describe only what is visible.\n\n"
    "Medium-class discipline:\n"
    "- Classify the medium as EXACTLY ONE of:\n"
    "    A. hand-drawn 2D — visible medium grain (paper tooth, watercolor bleeds), hand-made line with pressure variation\n"
    "    B. vector / flat 2D — uniform solid fills, bezier curves, zero grain, hard silhouette edges\n"
    "    C. stylized 3D CGI — rounded bevels, subdivision-smooth gradients, ambient occlusion in creases, rim light, matte plastic/fondant surfaces, no outlines\n"
    "    D. photoreal 3D CGI — PBR materials, physically plausible lighting, DoF bokeh, subsurface scattering, film grain, HDRI reflections\n"
    "    E. mixed / 2.5D — 2D assets with parallax, or 3D model with hand-painted textures\n"
    "- Use class-appropriate vocabulary ONLY. Never say 'brushstrokes' for C/D. Never say 'ambient occlusion' or "
    "'subsurface scattering' for A/B. Never say 'cel shading' for C unless you also see hand-drawn ink lines.\n"
    "- Correct 3D/CGI vocabulary: bevel, subdivision-smooth, ambient occlusion (AO), rim light, specular highlight, "
    "global illumination (GI), subsurface scattering (SSS), PBR roughness, matte plastic, fondant surface.\n"
    "- Correct 2D vocabulary: impasto, pigment pooling, paper tooth, ink bleed, pencil hatching, watercolor wash, "
    "halftone (hand-drawn); uniform fills, bezier curves, stepped gradient, hard silhouette (vector).\n\n"
    "Character-proportion discipline:\n"
    "- For every named character, emit head-heights-tall (numeric, e.g. '3.2 heads tall') AND an archetype phrase "
    "(chibi / stylized-youth / heroic / realistic-adult / elongated).\n"
    "- Describe silhouette primitives: dominant shape of torso (teardrop, cylinder, triangle), head (sphere, "
    "squashed sphere, egg), each limb (cylinder, tapered wedge, stubby nub).\n"
    "- Name observable body ratios when visible: head-to-shoulder width, torso-to-leg length, eye-width-to-face-width, "
    "hand-to-face size. Do not invent unobservable numbers.\n\n"
    "Way-of-drawing discipline:\n"
    "- Describe the construction recipe at the top of [Art Style]: silhouette primitives, construction order "
    "(head → torso → limbs → props), line policy (none / thin uniform / variable ink / painterly / crease-only), "
    "shading layers in order (base → AO → midtones → rim → specular), edge softness, texture grain, and one "
    "signature rendering quirk.\n\n"
    "Audit-block discipline (CITE, DO NOT RESTATE):\n"
    "- The meta-prompt defines named rules (Bevel Rule, Cool Rim Rule, AO-First Depth Rule, Fondant Surface Rule, "
    "Restricted Specular Rule, Chromatic Saturation Rule, Telephoto Readability Rule, Medium-Class Rule, etc.) "
    "exactly once in its ## style_foundation block.\n"
    "- In every downstream audit block ([Medium Class Verification], [Geometry Primitives], [Shading Layers], "
    "[Ambient Occlusion Map], [Rim Light Policy], [Specular Allocation], [Surface Cleanliness], [Camera & Depth], "
    "and any analogous audit block under a different schema), cite rule names by reference (e.g. 'Bevel Rule applies "
    "to the hero mass; the outer contour is rounded and beveled', 'Cool Rim Rule permits a thin cyan rim along the "
    "upper silhouette only').\n"
    "- Do NOT paraphrase or re-state rule content. Do NOT open an audit block with a rule definition. Every sentence "
    "in an audit block must point to image-specific evidence or to a named rule already defined upstream.\n\n"
    "Output-format discipline:\n"
    "- Write section labels as PLAIN TEXT: `[Art Style]`, `[Subject]`. Never wrap them in markdown bolding "
    "(`**[Art Style]**`), backticks, angle brackets, or any other formatting.\n"
    "- Emit the labeled blocks in the exact order the meta-prompt's ## Caption Sections block specifies. "
    "Target the word count in its ## Caption Length Target block; let [Art Style] and [Subject] carry most of "
    "the word budget, and keep audit blocks lean (image evidence, not rule recap).\n\n"
    "Forbidden terms (never use unless quoting a named art movement):\n"
    "  cartoon, cartoonish, stylised, stylized (as bare adjective), beautiful, epic, whimsical, charming, cinematic."
)

CAPTION_PROMPT = (
    "Describe this image in comprehensive detail for someone who cannot see it. "
    "These descriptions will be used to understand and reproduce the art style.\n\n"
    "## Output format\n"
    "Use these labeled sections in this exact order. Keep the labels exactly as written and let each section expand "
    "to the length needed for concrete visual detail.\n\n"
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
    "- Target 2000-6000 words total.\n"
    "- Do not speculate about the artist's intent; describe only what is visible.\n"
    "- [Subject] and [Art Style] each 800-2000 words when the image supports that detail; they should be the longest sections.\n"
    "- Ancillary sections should usually land in the 150-400 word range while staying concrete.\n"
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

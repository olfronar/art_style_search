"""Zero-step style analysis: build a StyleProfile and initial PromptTemplate from reference images."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from google import genai

from art_style_search.types import Caption, PromptSection, PromptTemplate, StyleProfile
from art_style_search.utils import (
    ReasoningClient,
    async_retry,
    extract_xml_tag,
    gemini_circuit_breaker,
    image_to_gemini_part,
)

logger = logging.getLogger(__name__)

_ANALYSIS_SYSTEM = (
    "You are an expert art analyst and prompt engineer. "
    "Your goal is to deeply understand an art style from reference images so it can be reproduced "
    "via text-to-image generation. Your analysis will directly feed into crafting generation prompts, "
    "so prioritize elements that are controllable via text prompts (color descriptions, technique keywords, "
    "mood descriptors) over elements that are hard to specify textually (exact spatial layouts, precise proportions)."
)

_GEMINI_ANALYSIS_PROMPT = (
    "Analyze the art style across ALL the provided reference images. "
    "You are the VISUAL specialist — focus on concrete visual specifics that you can see directly:\n\n"
    "1. **Exact colors**: Name specific colors (e.g. 'burnt sienna', 'cadmium yellow'), dominant palette, "
    "color temperature, saturation levels, gradient patterns.\n"
    "2. **Rendering technique**: Visible brushstrokes or rendering artifacts, line quality, edge treatment, "
    "level of detail vs abstraction, texture of the medium.\n"
    "3. **Spatial composition**: How elements are arranged, recurring framing patterns, use of negative space, "
    "depth and perspective conventions.\n"
    "4. **Light and color relationships**: How light behaves, shadow treatment, color harmonies, "
    "contrast levels, atmospheric effects.\n"
    "5. **Surface textures**: Visual texture patterns, how materials are rendered, grain or noise characteristics.\n"
    "6. **Distinctive visual signatures**: Anything visually unique that would distinguish this style — "
    "unusual color combinations, characteristic marks, recurring visual motifs.\n\n"
    "Be extremely specific about what you SEE. Use precise color names, describe exact visual qualities. "
    "Identify patterns across MULTIPLE images. This feeds into crafting a text-to-image prompt."
)

_REASONING_ANALYSIS_PROMPT = (
    "Below are detailed text descriptions of reference images that share a common art style. "
    "You are the REASONING specialist — focus on abstract patterns, underlying principles, and stylistic rules.\n\n"
    "{captions}\n\n"
    "Analyze the descriptions to identify:\n"
    "1. **Stylistic rules**: What consistent principles govern this style? What would an artist following "
    "this style always do or never do?\n"
    "2. **Technique classification**: What medium, art movement, or historical period does this most closely "
    "resemble? What specific techniques define it?\n"
    "3. **Mood and emotional logic**: What emotional tone unifies the works? How do color/composition choices "
    "serve the mood?\n"
    "4. **What makes it distinctive**: How would you distinguish this from similar styles? What's the 'signature'?\n"
    "5. **Prompt-critical elements**: Which aspects are most important to specify in a text-to-image prompt "
    "to reproduce this style? Rank by importance.\n"
    "6. **Common pitfalls**: What would a generic AI image generator likely get WRONG about this style "
    "without specific guidance?\n\n"
    "Focus on insights that help write better generation prompts. Identify cross-image patterns, not one-offs. "
    "Think about what text descriptions would be most effective at reproducing this style."
)

_COMPILATION_PROMPT = (
    "You received two independent analyses of the same set of reference images.\n\n"
    "## Analysis from visual model (Gemini — saw the actual images):\n{gemini_analysis}\n\n"
    "## Analysis from reasoning model (read detailed captions):\n{reasoning_analysis}\n\n"
    "Synthesize BOTH analyses into:\n"
    "1. A structured **StyleProfile** with 6 sections.\n"
    "2. An initial **PromptTemplate** — a set of prompt sections that instruct a captioner "
    "(Gemini Pro) HOW to describe reference images. The captions produced must serve a dual "
    "purpose: (a) contain enough detail to recreate the image, and (b) embed reusable "
    "art-style guidance in labeled sections that can later generate NEW art in the same style.\n\n"
    "Where the two analyses disagree, prefer the visual model's observations for concrete visual details "
    "(colors, textures) and the reasoning model's insights for abstract principles (style rules, mood logic).\n\n"
    "**Meta-prompt engineering guidance:**\n"
    "- Use specific, descriptive language — concrete color names, technique terms, art-movement references.\n"
    "- Front-load the most distinctive style elements.\n"
    "- Create a COMPREHENSIVE template with 8-15 sections covering ALL visual aspects.\n"
    "- Each section value should be detailed (4-8 sentences) and EMBED the core style rules "
    "from the StyleProfile as literal text that the captioner weaves into every caption.\n"
    "- Total rendered meta-prompt should be 1200-1800 words.\n"
    "- The negative prompt should target common failure modes the reasoning model identified.\n"
    "- Include <caption_sections>: an ordered comma-separated list of labeled sections "
    "the captioner should produce in its output (e.g. Art Style, Color Palette, Composition, etc.).\n"
    "- Include <caption_length>: target word count for produced captions (e.g. 500).\n\n"
    "The prompt template should have 8-15 sections covering: technique/medium, color palette, composition, "
    "character/figure treatment, background/environment, details/textures, lighting, mood/atmosphere, "
    "and optionally: art style overview, subject rendering, emotional tone. "
    "Each section should have a short name, a description of what it controls, "
    "and detailed prompt text with embedded style rules as its value. Include a thorough negative prompt.\n\n"
    "Respond in EXACTLY this XML format (no markdown fences, no extra text outside the XML):\n"
    "<style_profile>\n"
    "  <color_palette>detailed description of the color palette</color_palette>\n"
    "  <composition>detailed description of composition patterns</composition>\n"
    "  <technique>detailed description of technique</technique>\n"
    "  <mood_atmosphere>detailed description of mood and atmosphere</mood_atmosphere>\n"
    "  <subject_matter>detailed description of subject matter</subject_matter>\n"
    "  <influences>detailed description of influences</influences>\n"
    "</style_profile>\n"
    "<initial_template>\n"
    '  <section name="art_style_overview" description="core art style identity and rules">'
    "detailed style identity with embedded rules (4-8 sentences)</section>\n"
    '  <section name="medium_and_technique" description="art medium, rendering style, brushwork">'
    "detailed technique description with style rules (4-8 sentences)</section>\n"
    '  <section name="color_palette" description="dominant colors and color relationships">'
    "detailed color description with palette rules (4-8 sentences)</section>\n"
    '  <section name="composition" description="layout, framing, spatial arrangement">'
    "detailed composition description with style patterns (4-8 sentences)</section>\n"
    '  <section name="characters" description="how figures and characters are rendered">'
    "detailed character treatment with style rules (4-8 sentences)</section>\n"
    '  <section name="background" description="environment and setting treatment">'
    "detailed background description with style rules (4-8 sentences)</section>\n"
    '  <section name="details" description="textures, patterns, fine elements">'
    "detailed texture and detail description (4-8 sentences)</section>\n"
    '  <section name="lighting" description="light source, quality, shadows">'
    "detailed lighting description with style rules (4-8 sentences)</section>\n"
    '  <section name="mood" description="emotional tone, atmosphere">'
    "detailed mood description with style rules (4-8 sentences)</section>\n"
    "  <negative>thorough list of things to avoid in generation</negative>\n"
    "  <caption_sections>Art Style, Color Palette, Technique, Composition, Characters, "
    "Background, Details, Lighting, Mood</caption_sections>\n"
    "  <caption_length>500</caption_length>\n"
    "</initial_template>"
)


# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------


_extract_tag = extract_xml_tag  # local alias for tests that import it


def _parse_sections(xml: str) -> list[PromptSection]:
    """Parse all <section name="..." description="...">value</section> tags."""
    import re

    pattern = r'<section\s+name="([^"]+)"\s+description="([^"]+)">(.*?)</section>'
    matches = re.findall(pattern, xml, re.DOTALL)
    return [
        PromptSection(name=name.strip(), description=desc.strip(), value=val.strip()) for name, desc, val in matches
    ]


def _parse_compilation(text: str, gemini_raw: str, reasoning_raw: str) -> tuple[StyleProfile, PromptTemplate]:
    """Parse the reasoning-model compilation response into StyleProfile + PromptTemplate."""
    profile = StyleProfile(
        color_palette=_extract_tag(text, "color_palette"),
        composition=_extract_tag(text, "composition"),
        technique=_extract_tag(text, "technique"),
        mood_atmosphere=_extract_tag(text, "mood_atmosphere"),
        subject_matter=_extract_tag(text, "subject_matter"),
        influences=_extract_tag(text, "influences"),
        gemini_raw_analysis=gemini_raw,
        claude_raw_analysis=reasoning_raw,
    )

    template_block = _extract_tag(text, "initial_template")
    sections = _parse_sections(template_block)
    negative = _extract_tag(template_block, "negative")

    caption_sections_raw = _extract_tag(template_block, "caption_sections")
    caption_sections = [s.strip() for s in caption_sections_raw.split(",") if s.strip()] if caption_sections_raw else []

    caption_length_raw = _extract_tag(template_block, "caption_length")
    caption_length_target = int(caption_length_raw) if caption_length_raw.isdigit() else 0

    template = PromptTemplate(
        sections=sections,
        negative_prompt=negative or None,
        caption_sections=caption_sections,
        caption_length_target=caption_length_target,
    )

    return profile, template


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


async def _gemini_analyze(
    reference_paths: list[Path],
    *,
    client: genai.Client,
    model: str,
) -> str:
    """Send all reference images to Gemini and get a style analysis."""
    contents: list[object] = []

    for img_path in reference_paths:
        contents.append(image_to_gemini_part(img_path))

    contents.append(_GEMINI_ANALYSIS_PROMPT)

    logger.info("Sending %d images to Gemini (%s) for style analysis", len(reference_paths), model)

    async def _call() -> str:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(model=model, contents=contents),
            timeout=120,
        )
        return response.text


    return await async_retry(_call, label="Gemini style analysis", circuit_breaker=gemini_circuit_breaker)


async def _reasoning_analyze(
    captions: list[Caption],
    *,
    client: ReasoningClient,
    model: str,
) -> str:
    """Send all captions to the reasoning model for style analysis."""
    captions_text = "\n\n---\n\n".join(f"**Image: {cap.image_path.name}**\n{cap.text}" for cap in captions)
    user = _REASONING_ANALYSIS_PROMPT.format(captions=captions_text)

    logger.info("Sending %d captions to %s for style analysis", len(captions), model)
    return await client.call(model=model, system=_ANALYSIS_SYSTEM, user=user, max_tokens=8000)


async def _reasoning_compile(
    gemini_analysis: str,
    reasoning_analysis: str,
    *,
    client: ReasoningClient,
    model: str,
) -> tuple[StyleProfile, PromptTemplate]:
    """Synthesize both analyses into a StyleProfile and PromptTemplate."""
    user = _COMPILATION_PROMPT.format(
        gemini_analysis=gemini_analysis,
        reasoning_analysis=reasoning_analysis,
    )

    logger.info("Compiling style profile via %s", model)
    raw_text = await client.call(model=model, system=_ANALYSIS_SYSTEM, user=user, max_tokens=12000)
    return _parse_compilation(raw_text, gemini_raw=gemini_analysis, reasoning_raw=reasoning_analysis)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _save_cache(profile: StyleProfile, template: PromptTemplate, cache_path: Path) -> None:
    """Persist style profile and template to disk."""
    from art_style_search.state import _to_dict

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "style_profile": _to_dict(profile),
        "prompt_template": _to_dict(template),
    }
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Style analysis cached to %s", cache_path)


def _load_cache(cache_path: Path) -> tuple[StyleProfile, PromptTemplate] | None:
    """Load cached style profile + template. Returns None on any failure."""
    if not cache_path.exists():
        return None

    try:
        from art_style_search.state import _prompt_template_from_dict, _style_profile_from_dict

        data = json.loads(cache_path.read_text(encoding="utf-8"))
        profile = _style_profile_from_dict(data["style_profile"])
        template = _prompt_template_from_dict(data["prompt_template"])
        logger.info("Loaded cached style analysis from %s", cache_path)
        return profile, template
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Corrupt style analysis cache at %s, will re-analyze: %s", cache_path, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def analyze_style(
    reference_paths: list[Path],
    captions: list[Caption],
    *,
    gemini_client: genai.Client,
    reasoning_client: ReasoningClient,
    caption_model: str,
    reasoning_model: str,
    cache_path: Path,
) -> tuple[StyleProfile, PromptTemplate]:
    """Perform zero-step style analysis: build a StyleProfile and initial PromptTemplate.

    1. Check cache — return early if valid.
    2. Run Gemini vision analysis and reasoning-model text analysis in parallel.
    3. Have the reasoning model compile both into structured outputs.
    4. Cache result to disk.
    """
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    # Run both analyses in parallel
    gemini_result, reasoning_result = await asyncio.gather(
        _gemini_analyze(reference_paths, client=gemini_client, model=caption_model),
        _reasoning_analyze(captions, client=reasoning_client, model=reasoning_model),
    )

    # Compile into structured output
    profile, template = await _reasoning_compile(
        gemini_result,
        reasoning_result,
        client=reasoning_client,
        model=reasoning_model,
    )

    # Cache to disk
    _save_cache(profile, template, cache_path)

    return profile, template

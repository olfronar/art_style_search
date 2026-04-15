"""Zero-step style analysis: build a StyleProfile and initial PromptTemplate from reference images."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from google import genai  # type: ignore[attr-defined]
from google.genai import types as genai_types  # type: ignore[attr-defined]

from art_style_search.prompt._parse import validate_template
from art_style_search.prompt.json_contracts import response_schema, schema_hint, validate_style_compilation_payload
from art_style_search.state import prompt_template_from_dict, style_profile_from_dict, to_dict
from art_style_search.types import Caption, PromptTemplate, StyleProfile
from art_style_search.utils import (
    ReasoningClient,
    async_retry,
    image_to_gemini_part,
    vision_circuit_breaker,
)

logger = logging.getLogger(__name__)

_ANALYSIS_SYSTEM = (
    "You are an expert art analyst and prompt engineer with strong art-historical vocabulary. "
    "Ground every claim in visible evidence, prefer concrete colors and technique terms over metaphor, "
    "and avoid speculating about artist intent or invisible process. "
    "Your analysis feeds prompt construction for text-to-image generation, so prioritize details that can be controlled with text."
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
    "Identify patterns across MULTIPLE images. This feeds into crafting a text-to-image prompt.\n"
    "Target 300-600 words total. Be specific but concise — avoid restating the same observation in multiple sections."
)

_REASONING_ANALYSIS_PROMPT = (
    "Below are detailed text descriptions of reference images that share a common art style.\n\n"
    "You are the REASONING specialist — focus on abstract patterns, underlying principles, and stylistic rules.\n\n"
    "{captions}\n\n"
    "Analyze the descriptions to identify:\n"
    "1. **Prompt-critical elements** (START HERE — most important): Which aspects are most important "
    "to specify in a text-to-image prompt to reproduce this style? Rank by importance.\n"
    "2. **Common pitfalls**: What would a generic AI image generator likely get WRONG about this style "
    "without specific guidance?\n"
    "3. **Stylistic rules**: What consistent principles govern this style? What would an artist following "
    "this style always do or never do?\n"
    "4. **Technique classification**: What medium, art movement, or historical period does this most closely "
    "resemble? What specific techniques define it?\n"
    "5. **Mood and emotional logic**: What emotional tone unifies the works? How do color/composition choices "
    "serve the mood?\n"
    "6. **What makes it distinctive**: How would you distinguish this from similar styles? What's the 'signature'?\n\n"
    "Focus on insights that help write better generation prompts. Identify cross-image patterns, not one-offs. "
    "Think about what text descriptions would be most effective at reproducing this style.\n"
    "Target 300-600 words total. Be specific but concise."
)

_COMPILATION_PROMPT = (
    "You received two independent analyses of the same set of reference images.\n"
    "Your template will define its OWN section names and structure based on the strongest cross-image patterns.\n\n"
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
    "- Use specific, descriptive language — concrete color names, technique terms, art-movement references. "
    "Why: generic terms like 'warm colors' produce vague captions; specific names like 'burnt sienna' "
    "produce images that match the reference.\n"
    "- Front-load the most distinctive style elements. "
    "Why: text-to-image models attend more to early prompt tokens.\n"
    "- Create a COMPREHENSIVE template with 8-20 sections covering ALL visual aspects.\n"
    "- Each section value should be detailed and EMBED the core style rules "
    "from the StyleProfile as literal text that the captioner weaves into every caption. "
    "Why: embedding rules as literal text ensures the captioner repeats them verbatim rather than interpreting them.\n"
    "- Total rendered meta-prompt should be 2000-8000 words.\n"
    "- The negative prompt should target common failure modes the reasoning model identified.\n"
    '- Include the JSON key "caption_sections": an ordered list of labeled sections '
    'the captioner should produce in its output (e.g. ["Art Style", "Subject", "Color Palette"]).\n'
    '- Include the JSON key "caption_length_target": target word count for produced captions (e.g. 4000).\n\n'
    "The prompt template should have 8-20 sections covering: technique/medium, color palette, composition, "
    "character/figure treatment, background/environment, details/textures, lighting, mood/atmosphere, "
    "and subject rendering. The FIRST section must be 'style_foundation' and the SECOND section must be "
    "'subject_anchor'. The first two caption output labels must be 'Art Style' and 'Subject'. "
    "The [Subject] block must require identity/species, distinguishing features, clothing or equipment, "
    "pose or action, expression, and props or context, targeting roughly 1000-2000 words, with [Art Style] also "
    "expected to run about 1000-2000 words and the remaining caption sections typically 150-400 words each. "
    "Each section should have a short name, a description of what it controls, "
    "and detailed prompt text with embedded style rules as its value. Include a thorough negative prompt.\n\n"
    "## Example (different art style — for format reference only, do NOT copy its content)\n"
    "This example shows the expected structure for a watercolor landscape style. "
    "Your output should match this structure but contain content specific to the analyzed style.\n\n"
    "```json\n"
    '{{"style_profile":{{"color_palette":"Cool blues and greens with occasional warm sienna accents, high transparency, water bloom effects","composition":"Rule-of-thirds landscape format with atmospheric perspective, soft horizon lines","technique":"Transparent watercolor on cold-pressed paper, wet-on-wet for skies, dry brush for texture","mood_atmosphere":"Serene, contemplative, misty morning light","subject_matter":"Natural landscapes with occasional architectural elements","influences":"English watercolor tradition, Turner atmospheric effects"}},"initial_template":{{"sections":[{{"name":"style_foundation","description":"core style rules","value":"..."}},{{"name":"subject_anchor","description":"subject fidelity","value":"..."}}],"negative_prompt":"...","caption_sections":["Art Style","Subject","Color Palette","Technique","Composition","Lighting & Atmosphere","Textures"],"caption_length_target":4000}}}}\n'
    "```\n\n"
    "## EXECUTION CHECKLIST — verify before outputting\n"
    "- [ ] style_profile has all 6 fields (color_palette, composition, technique, mood_atmosphere, subject_matter, influences)\n"
    "- [ ] Template first section is 'style_foundation', second is 'subject_anchor'\n"
    "- [ ] caption_sections starts with ['Art Style', 'Subject', ...]\n"
    "- [ ] Template has 8-20 sections\n"
    "- [ ] Negative prompt is included\n\n"
    "Respond with EXACTLY one JSON object (no markdown fences, no extra text):\n"
    "{{\n"
    '  "style_profile": {{\n'
    '    "color_palette": "...",\n'
    '    "composition": "...",\n'
    '    "technique": "...",\n'
    '    "mood_atmosphere": "...",\n'
    '    "subject_matter": "...",\n'
    '    "influences": "..."\n'
    "  }},\n"
    '  "initial_template": {{\n'
    '    "sections": [{{"name": "style_foundation", "description": "core art style identity and rules", "value": "..."}}, {{"name": "subject_anchor", "description": "subject fidelity instructions", "value": "..."}}, {{"name": "color_palette", "description": "palette guidance", "value": "..."}}, {{"name": "composition", "description": "layout guidance", "value": "..."}}],\n'
    '    "negative_prompt": "...",\n'
    '    "caption_sections": ["Art Style", "Subject", "Color Palette", "Technique", "Composition"],\n'
    '    "caption_length_target": 4000\n'
    "  }}\n"
    "}}"
)


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
            client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=genai_types.GenerateContentConfig(system_instruction=_ANALYSIS_SYSTEM),
            ),
            timeout=120,
        )
        return response.text

    return await async_retry(_call, label="Gemini style analysis", circuit_breaker=vision_circuit_breaker)


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
    return await client.call_json(
        model=model,
        system=_ANALYSIS_SYSTEM,
        user=user,
        validator=lambda data: validate_style_compilation_payload(
            data,
            gemini_raw=gemini_analysis,
            reasoning_raw=reasoning_analysis,
        ),
        response_name="style_compilation",
        schema_hint=schema_hint("style_compilation"),
        response_schema=response_schema("style_compilation"),
        max_tokens=20000,
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _save_cache(profile: StyleProfile, template: PromptTemplate, cache_path: Path) -> None:
    """Persist style profile and template to disk."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "style_profile": to_dict(profile),
        "prompt_template": to_dict(template),
    }
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Style analysis cached to %s", cache_path)


def _load_cache(cache_path: Path) -> tuple[StyleProfile, PromptTemplate] | None:
    """Load cached style profile + template. Returns None on any failure."""
    if not cache_path.exists():
        return None

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        profile = style_profile_from_dict(data["style_profile"])
        template = prompt_template_from_dict(data["prompt_template"])
        template_errors = validate_template(template)
        if template_errors:
            logger.warning(
                "Invalid style analysis cache at %s, will re-analyze: %s",
                cache_path,
                "; ".join(template_errors),
            )
            return None
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

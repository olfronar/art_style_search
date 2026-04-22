"""Zero-step style analysis: build a StyleProfile and initial PromptTemplate from reference images."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from google import genai  # type: ignore[attr-defined]
from google.genai import types as genai_types  # type: ignore[attr-defined]

from art_style_search.prompt._parse import validate_template
from art_style_search.prompt.json_contracts import response_schema, schema_hint, validate_style_compilation_payload
from art_style_search.reasoning_client import ANTHROPIC_EFFORT_FROM_THINKING
from art_style_search.state import prompt_template_from_dict, style_profile_from_dict, to_dict
from art_style_search.types import Caption, PromptTemplate, StyleProfile
from art_style_search.utils import (
    ReasoningClient,
    async_retry,
    gemini_timeout_s,
    image_to_gemini_part,
    vision_circuit_breaker,
)

# Gemini analysis returns a ~8000-token structured analysis — drives timeout scaling.
_ANALYSIS_MAX_OUTPUT_TOKENS = 8000
_ANALYSIS_TIMEOUT_S = gemini_timeout_s(_ANALYSIS_MAX_OUTPUT_TOKENS)

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
    "0. **Medium identification** (answer FIRST — this anchors every downstream decision). "
    "Describe the medium in 2-3 sentences using plain, observable vocabulary that matches the specific surface "
    "you see. What behavior does the surface actually exhibit — the grain of marks, the edge treatment, the way "
    "form is built, how light lands? Cite 3-5 concrete observable cues and let them drive your word choice. "
    "Avoid genre labels and avoid picking from any implicit menu — the vocabulary should emerge from the "
    "observation, not from a checklist. If references split across media, report primary + secondary.\n"
    "1. **Exact colors**: Name specific colors (e.g. 'burnt sienna', 'cadmium yellow'), dominant palette, "
    "color temperature, saturation levels, gradient patterns.\n"
    "2. **Rendering technique**: Use vocabulary that matches what you observed in item 0. "
    "Describe line quality, edge treatment, level of detail vs abstraction, and the character of the medium's "
    "texture. Pick terms that match the actual surface behavior. Don't mix vocabulary across incompatible media; "
    "let the observed surface drive word choice.\n"
    "3. **Spatial composition**: How elements are arranged, recurring framing patterns, use of negative space, "
    "depth and perspective conventions.\n"
    "4. **Light and color relationships**: How light behaves, shadow treatment, color harmonies, "
    "contrast levels, atmospheric effects.\n"
    "5. **Surface textures**: Visual texture patterns, how materials are rendered, grain or noise characteristics.\n"
    "6. **Character proportions** (when characters are visible): estimate head-heights-tall across the cast "
    "(e.g. '3 heads tall / chibi', '5 heads / stylized-youth', '7.5 heads / realistic-adult', 'elongated 8+'). "
    "Note dominant silhouette primitives (torso shape, head shape, limb shape) and eye-to-face ratio.\n"
    "7. **Distinctive visual signatures**: Anything visually unique that would distinguish this style — "
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
    "4. **Technique principles**: What specific techniques define this style? Describe finished-surface "
    "characteristics — line/edge policy, shading stack, surface vocabulary, palette principle. Do NOT narrate "
    "drawing procedure (construction steps, primitives, fabrication, how an artist would build the image). "
    "Do NOT name a genre, movement, or period; principles travel, labels don't.\n"
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
    "- **'style_foundation.value' IS the [Art Style] canon.** It is the literal declarative text the captioner "
    "pastes verbatim into every caption's [Art Style] block AND the text the image generator reads as the style "
    "descriptor. Write it as third-person assertions ABOUT this style ('This style renders as...', 'Edges are...', "
    "'The palette pulls from...'), NOT as directives to a captioner. Forbidden inside the value: imperative verbs "
    "addressed to a reader (Begin, Write, Declare, Target, Avoid, Follow, Do not, Before), audit scaffolding "
    "(SLOT N, '- [ ]', MANDATORY), meta-references ('this block', 'each slot', 'REUSABLE DNA'), and word-count "
    "targets. Every sentence you put in the value must still be true of a DIFFERENT image in this style — "
    "per-image observations (specific body parts, named objects, proper nouns, actual colors, pose details) "
    "belong in [Subject], [Color Palette], [Composition], or [Lighting & Atmosphere], NEVER in [Art Style].\n"
    "- Be laconic: state each style assertion ONCE in 'style_foundation'. Ancillary sections describe per-image "
    "observations without restating the canon. Duplication inflates rendered word count without adding signal.\n"
    "- 'style_foundation.value' covers five facets in order, each 2-4 declarative sentences describing this style. "
    "The vocabulary for every facet must emerge from what the two analyses above actually surfaced — do NOT "
    "reach for words the analyses did not use, and do NOT pull from an implicit menu. Use the analyses' own "
    "surface observations (grain, line, edge behavior, shading response, material response) as the vocabulary "
    "source; if the analyses did not name a particular rendering technique, do not invent one to fill a slot:\n"
    "    1. How to Draw (open with the literal marker 'How to Draw:' on its own sub-line). Describe the "
    "**finished surface** in plain observable vocabulary matching what the two analyses reported — not "
    "construction steps, primitives, or fabrication. Write declarative sentences about the medium's surface "
    "behavior and line/edge policy (how contours resolve in the finished image); do NOT narrate how an artist "
    "would build the image from shapes or stages.\n"
    "    2. Shading & Light. Write declarative sentences that name the shading layers, the edge-softness "
    "character, and the key/fill/rim direction and temperature the analyses actually observed in these references. "
    "No body parts, no specific objects.\n"
    "    3. Color Principle. Write declarative sentences that name the palette families, value range, saturation "
    "policy, and shadow-direction rule observed in these references — in generic terms (families, not hex codes), "
    "without naming image-specific hues.\n"
    "    4. Surface & Texture. Write declarative sentences that name the grain/noise policy and the "
    "class-appropriate material vocabulary observed in these references, using words self-consistent with the "
    "medium just named in How to Draw; no specific objects.\n"
    "    5. Style Invariants. Write 3-5 MUST/NEVER rules that every image in this style obeys, each a standalone "
    "sentence, phrased as generative claims drawn directly from the patterns the analyses identified.\n"
    "- Character proportions (heads-tall + archetype) belong in subject_anchor, NOT in style_foundation.\n"
    "- Genre labels are forbidden inside 'style_foundation.value' — '3D CGI of X', 'cel-shaded anime', "
    "'{{Artist}}-style', 'watercolor illustration' must not appear. Describe the technique so a reader "
    "reconstructs the genre from the principles.\n"
    "- 'subject_anchor.value' describes how the captioner should write the per-image [Subject] block and MUST "
    "contain a 'Proportions:' sub-block requiring the captioner to emit head-heights-tall (numeric, e.g. "
    "'3.2 heads tall') AND an archetype phrase — one of: chibi / stylized-youth / heroic / realistic-adult / "
    "elongated. This anchors character anatomy against generator drift toward default 7.5-head realism.\n"
    "- Total rendered meta-prompt should be 2000-8000 words.\n"
    "- The negative prompt should target common failure modes the reasoning model identified.\n"
    '- Include the JSON key "caption_sections": an ordered list of labeled sections '
    'the captioner should produce in its output (e.g. ["Art Style", "Subject", "Color Palette"]).\n'
    '- Include the JSON key "caption_length_target": target word count for produced captions (e.g. 4000).\n\n'
    "The prompt template should have 8-20 sections covering: color palette, composition, "
    "character/figure treatment, background/environment, lighting, mood/atmosphere, "
    "and subject rendering. The FIRST section must be 'style_foundation' and the SECOND section must be "
    "'subject_anchor'. The first two caption output labels must be 'Art Style' and 'Subject'. "
    "The [Subject] block must require identity/species, distinguishing features, clothing or equipment, "
    "pose or action, expression, props or context, and character proportions (heads-tall + archetype), "
    "targeting roughly 800-2000 words. [Art Style] is the style_foundation canon pasted verbatim — "
    "declarative assertions about this style, typically 400-800 words across the 5 facets. "
    "Ancillary caption sections typically 150-400 words each. "
    "By default, do NOT include 'Technique' or 'Textures' as caption_sections — their content duplicates [Art Style]; "
    "propose them as experiments only if you have a specific mechanism-level reason.\n"
    "Each section should have a short name, a description of what it controls, and detailed prompt text as its value. "
    "Include a thorough negative prompt.\n\n"
    "## Schema (structural placeholders only — do NOT copy this vocabulary; fill with content "
    "specific to the analyzed style based on the two analyses above)\n"
    "```json\n"
    '{{"style_profile":{{"color_palette":"[palette families + value + saturation policy observed]","composition":"[recurring composition + framing patterns observed]","technique":"[rendering technique + surface behavior observed]","mood_atmosphere":"[consistent mood/atmosphere observed]","subject_matter":"[recurring subject matter observed]","influences":"[principle-level influences; no genre or artist labels]"}},"initial_template":{{"sections":[{{"name":"style_foundation","description":"the literal [Art Style] canon — copied verbatim into every caption and read by the image generator as the style descriptor","value":"[canon body covering How to Draw / Shading & Light / Color Principle / Surface & Texture / Style Invariants]"}},{{"name":"subject_anchor","description":"subject fidelity instructions","value":"[subject block structure requiring Proportions: + archetype]"}}],"negative_prompt":"[concrete failure modes observed]","caption_sections":["Art Style","Subject","Color Palette","Composition","Lighting & Atmosphere"],"caption_length_target":3000}}}}\n'
    "```\n\n"
    "## EXECUTION CHECKLIST — verify before outputting\n"
    "- [ ] style_profile has all 6 fields (color_palette, composition, technique, mood_atmosphere, subject_matter, influences)\n"
    "- [ ] Template first section is 'style_foundation', second is 'subject_anchor'\n"
    "- [ ] caption_sections starts with ['Art Style', 'Subject', ...]\n"
    "- [ ] Template has 8-20 sections\n"
    "- [ ] style_foundation.value reads as third-person declarative assertions ABOUT this style "
    "('This style renders as...', 'Edges are...'), NOT as imperatives addressed to a captioner\n"
    "- [ ] style_foundation.value does NOT contain the strings 'Begin the caption', 'Each slot', "
    "'This block is', 'Do not', 'Write', 'MANDATORY', 'SLOT', '- [ ]', or any 'N-M words' word-count reference\n"
    "- [ ] style_foundation.value covers the five facets in order (How to Draw / Shading & Light / "
    "Color Principle / Surface & Texture / Style Invariants), opening with the literal 'How to Draw:' marker\n"
    "- [ ] style_foundation.value contains no image-specific nouns, no proper nouns, no named colors — "
    "every sentence would still be true of a different image in this style\n"
    "- [ ] caption_sections does NOT include 'Technique' or 'Textures' by default\n"
    "- [ ] subject_anchor.value contains a 'Proportions:' sub-block AND at least one archetype token "
    "(chibi / stylized-youth / heroic / realistic-adult / elongated)\n"
    "- [ ] style_foundation identifies the medium in plain observable vocabulary (no bucketed letter class) "
    "and its downstream technique language stays consistent with what was observed\n"
    "- [ ] Each style assertion is stated ONCE (no verbatim duplication across sections)\n"
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
    '    "sections": [{{"name": "style_foundation", "description": "the literal [Art Style] canon — copied verbatim into every caption and read by the image generator as the style descriptor", "value": "..."}}, {{"name": "subject_anchor", "description": "subject fidelity instructions", "value": "..."}}, {{"name": "color_palette", "description": "palette guidance", "value": "..."}}, {{"name": "composition", "description": "layout guidance", "value": "..."}}],\n'
    '    "negative_prompt": "...",\n'
    '    "caption_sections": ["Art Style", "Subject", "Color Palette", "Composition", "Lighting & Atmosphere"],\n'
    '    "caption_length_target": 3000\n'
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
    contents: list[Any] = []

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
            timeout=_ANALYSIS_TIMEOUT_S,
        )
        return response.text or ""

    return await async_retry(_call, label="Gemini style analysis", circuit_breaker=vision_circuit_breaker)


async def _claude_visual_analyze(
    reference_paths: list[Path],
    *,
    client: ReasoningClient,
    model: str,
    thinking_level: str = "MINIMAL",
) -> str:
    """Send all reference images to an Anthropic-compatible reasoning client for style analysis.

    Wraps the underlying call in :func:`async_retry` + ``vision_circuit_breaker`` for parity
    with :func:`_gemini_analyze` so transient 429/500s don't fail zero-step on a single blip.
    """
    logger.info("Sending %d images to %s (%s) for visual style analysis", len(reference_paths), client.provider, model)
    effort = ANTHROPIC_EFFORT_FROM_THINKING.get(thinking_level.upper(), "low")

    async def _call() -> str:
        return await client.call_with_images(
            model=model,
            system=_ANALYSIS_SYSTEM,
            user=_GEMINI_ANALYSIS_PROMPT,
            image_paths=reference_paths,
            max_tokens=_ANALYSIS_MAX_OUTPUT_TOKENS,
            reasoning_effort=effort,
            stage="visual_analyze",
        )

    return await async_retry(_call, label="Claude visual analysis", circuit_breaker=vision_circuit_breaker)


async def _visual_analyze(
    reference_paths: list[Path],
    *,
    provider: str,
    gemini_client: genai.Client,
    gemini_model: str,
    reasoning_client: ReasoningClient,
    bootstrap_model: str,
    thinking_level: str = "MINIMAL",
) -> str:
    """Dispatch zero-step visual analysis to Gemini (default) or an Anthropic reasoning client."""
    if provider == "claude":
        if not bootstrap_model:
            msg = "visual_provider='claude' requires a non-empty bootstrap_model (e.g. 'claude-opus-4-7')"
            raise ValueError(msg)
        return await _claude_visual_analyze(
            reference_paths,
            client=reasoning_client,
            model=bootstrap_model,
            thinking_level=thinking_level,
        )
    return await _gemini_analyze(reference_paths, client=gemini_client, model=gemini_model)


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
    return await client.call(
        model=model,
        system=_ANALYSIS_SYSTEM,
        user=user,
        max_tokens=8000,
        temperature=0.3,
        stage="analyze",
    )


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
        temperature=0.3,
        stage="compile",
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
    visual_provider: str = "gemini",
    visual_model: str = "",
    visual_thinking_level: str = "MINIMAL",
) -> tuple[StyleProfile, PromptTemplate]:
    """Perform zero-step style analysis: build a StyleProfile and initial PromptTemplate.

    1. Check cache — return early if valid.
    2. Run visual analysis (``visual_provider``: gemini or claude) and reasoning-model text
       analysis in parallel.
    3. Have the reasoning model compile both into structured outputs.
    4. Cache result to disk.

    When ``visual_provider == "claude"``, the vision analysis routes through
    ``reasoning_client.call_with_images`` at ``visual_model`` (typically ``claude-opus-4-7``),
    with ``visual_thinking_level`` mapped to Anthropic reasoning effort.
    """
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    # Run both analyses in parallel
    visual_result, reasoning_result = await asyncio.gather(
        _visual_analyze(
            reference_paths,
            provider=visual_provider,
            gemini_client=gemini_client,
            gemini_model=caption_model,
            reasoning_client=reasoning_client,
            bootstrap_model=visual_model,
            thinking_level=visual_thinking_level,
        ),
        _reasoning_analyze(captions, client=reasoning_client, model=reasoning_model),
    )

    # Compile into structured output
    profile, template = await _reasoning_compile(
        visual_result,
        reasoning_result,
        client=reasoning_client,
        model=reasoning_model,
    )

    # Cache to disk
    _save_cache(profile, template, cache_path)

    return profile, template

"""Zero-step style analysis: build a StyleProfile and initial PromptTemplate from reference images."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

import anthropic
from google import genai

from art_style_search.types import Caption, PromptSection, PromptTemplate, StyleProfile
from art_style_search.utils import extract_text, image_to_gemini_part, stream_message

logger = logging.getLogger(__name__)

_ANALYSIS_SYSTEM = (
    "You are an expert art analyst and prompt engineer. "
    "Your goal is to deeply understand an art style from reference images so it can be reproduced via text-to-image generation."
)

_GEMINI_ANALYSIS_PROMPT = (
    "Analyze the art style across ALL the provided reference images. "
    "Focus on what makes this style UNIQUE and REPRODUCIBLE. Cover:\n"
    "1. **Color palette**: Dominant colors, relationships, temperature, saturation patterns.\n"
    "2. **Composition**: Recurring layout patterns, use of space, perspective, framing conventions.\n"
    "3. **Technique**: Medium, brushwork/rendering style, level of detail, abstraction level.\n"
    "4. **Mood & atmosphere**: Emotional tone, lighting patterns, sense of time/place.\n"
    "5. **Subject matter**: Common themes, recurring elements, how subjects are treated.\n"
    "6. **Influences**: Art movements, historical periods, or artists this style evokes.\n\n"
    "Be specific and concrete. Identify patterns that appear across MULTIPLE images, not just one. "
    "This analysis will be used to craft a text prompt that reproduces this exact style."
)

_CLAUDE_ANALYSIS_PROMPT = (
    "Below are detailed text descriptions of reference images that share a common art style. "
    "Analyze these descriptions to identify the DEFINING characteristics of the style.\n\n"
    "{captions}\n\n"
    "Focus on what makes this style UNIQUE and REPRODUCIBLE. Cover:\n"
    "1. **Color palette**: Dominant colors, relationships, temperature, saturation patterns.\n"
    "2. **Composition**: Recurring layout patterns, use of space, perspective, framing conventions.\n"
    "3. **Technique**: Medium, brushwork/rendering style, level of detail, abstraction level.\n"
    "4. **Mood & atmosphere**: Emotional tone, lighting patterns, sense of time/place.\n"
    "5. **Subject matter**: Common themes, recurring elements, how subjects are treated.\n"
    "6. **Influences**: Art movements, historical periods, or artists this style evokes.\n\n"
    "Be specific and concrete. Identify patterns that appear across MULTIPLE descriptions, not just one. "
    "This analysis will be used to craft a text prompt that reproduces this exact style."
)

_COMPILATION_PROMPT = (
    "You received two independent analyses of the same set of reference images.\n\n"
    "## Analysis from visual model (Gemini — saw the actual images):\n{gemini_analysis}\n\n"
    "## Analysis from text model (Claude — read detailed captions):\n{claude_analysis}\n\n"
    "Synthesize BOTH analyses into:\n"
    "1. A structured **StyleProfile** with 6 sections.\n"
    "2. An initial **PromptTemplate** — a set of prompt sections that, when concatenated, "
    "would instruct an image generation model to reproduce this style.\n\n"
    "The prompt template should have 3-6 sections, each with a short name, a description of what the section controls, "
    "and the actual prompt text as its value. Also include a negative prompt of things to avoid.\n\n"
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
    '  <section name="section_name" description="what this section controls">prompt text for this section</section>\n'
    '  <section name="another_section" description="what this controls">prompt text</section>\n'
    "  <negative>things to avoid in generation</negative>\n"
    "</initial_template>"
)


# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------


def _extract_tag(xml: str, tag: str) -> str:
    """Extract text content between <tag> and </tag>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, xml, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _parse_sections(xml: str) -> list[PromptSection]:
    """Parse all <section name="..." description="...">value</section> tags."""
    pattern = r'<section\s+name="([^"]+)"\s+description="([^"]+)">(.*?)</section>'
    matches = re.findall(pattern, xml, re.DOTALL)
    return [
        PromptSection(name=name.strip(), description=desc.strip(), value=val.strip()) for name, desc, val in matches
    ]


def _parse_compilation(text: str, gemini_raw: str, claude_raw: str) -> tuple[StyleProfile, PromptTemplate]:
    """Parse the Claude compilation response into StyleProfile + PromptTemplate."""
    profile = StyleProfile(
        color_palette=_extract_tag(text, "color_palette"),
        composition=_extract_tag(text, "composition"),
        technique=_extract_tag(text, "technique"),
        mood_atmosphere=_extract_tag(text, "mood_atmosphere"),
        subject_matter=_extract_tag(text, "subject_matter"),
        influences=_extract_tag(text, "influences"),
        gemini_raw_analysis=gemini_raw,
        claude_raw_analysis=claude_raw,
    )

    template_block = _extract_tag(text, "initial_template")
    sections = _parse_sections(template_block)
    negative = _extract_tag(template_block, "negative")

    template = PromptTemplate(
        sections=sections,
        negative_prompt=negative or None,
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
    response = await client.aio.models.generate_content(model=model, contents=contents)
    return response.text


async def _claude_analyze(
    captions: list[Caption],
    *,
    client: anthropic.AsyncAnthropic,
    model: str,
) -> str:
    """Send all captions to Claude and get a style analysis."""
    captions_text = "\n\n---\n\n".join(f"**Image: {cap.image_path.name}**\n{cap.text}" for cap in captions)
    prompt = _CLAUDE_ANALYSIS_PROMPT.format(captions=captions_text)

    logger.info("Sending %d captions to Claude (%s) for style analysis", len(captions), model)
    response = await stream_message(
        client,
        model=model,
        max_tokens=80000,
        thinking={"type": "adaptive"},
        system=_ANALYSIS_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_text(response)


async def _claude_compile(
    gemini_analysis: str,
    claude_analysis: str,
    *,
    client: anthropic.AsyncAnthropic,
    model: str,
) -> tuple[StyleProfile, PromptTemplate]:
    """Synthesize both analyses into a StyleProfile and PromptTemplate."""
    prompt = _COMPILATION_PROMPT.format(
        gemini_analysis=gemini_analysis,
        claude_analysis=claude_analysis,
    )

    logger.info("Compiling style profile via Claude (%s)", model)
    response = await stream_message(
        client,
        model=model,
        max_tokens=80000,
        thinking={"type": "adaptive"},
        system=_ANALYSIS_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = extract_text(response)
    return _parse_compilation(raw_text, gemini_raw=gemini_analysis, claude_raw=claude_analysis)


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
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        profile = StyleProfile(
            color_palette=data["style_profile"]["color_palette"],
            composition=data["style_profile"]["composition"],
            technique=data["style_profile"]["technique"],
            mood_atmosphere=data["style_profile"]["mood_atmosphere"],
            subject_matter=data["style_profile"]["subject_matter"],
            influences=data["style_profile"]["influences"],
            gemini_raw_analysis=data["style_profile"]["gemini_raw_analysis"],
            claude_raw_analysis=data["style_profile"]["claude_raw_analysis"],
        )
        sp = data["prompt_template"]
        template = PromptTemplate(
            sections=[
                PromptSection(name=s["name"], description=s["description"], value=s["value"])
                for s in sp.get("sections", [])
            ],
            negative_prompt=sp.get("negative_prompt"),
        )
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
    anthropic_client: anthropic.AsyncAnthropic,
    caption_model: str,
    claude_model: str,
    cache_path: Path,
) -> tuple[StyleProfile, PromptTemplate]:
    """Perform zero-step style analysis: build a StyleProfile and initial PromptTemplate.

    1. Check cache — return early if valid.
    2. Run Gemini vision analysis and Claude text analysis in parallel.
    3. Have Claude compile both into structured outputs.
    4. Cache result to disk.
    """
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    # Run both analyses in parallel
    gemini_result, claude_result = await asyncio.gather(
        _gemini_analyze(reference_paths, client=gemini_client, model=caption_model),
        _claude_analyze(captions, client=anthropic_client, model=claude_model),
    )

    # Compile into structured output
    profile, template = await _claude_compile(
        gemini_result,
        claude_result,
        client=anthropic_client,
        model=claude_model,
    )

    # Cache to disk
    _save_cache(profile, template, cache_path)

    return profile, template

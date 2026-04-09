"""Formatting helpers — render dataclasses into text blocks for the reasoning model."""

from __future__ import annotations

from art_style_search.types import AggregatedMetrics, PromptTemplate, StyleProfile


def _format_style_profile(profile: StyleProfile, compact: bool = False) -> str:
    """Render a StyleProfile into a text block for system prompts.

    When *compact* is True, omits the raw analyses to save tokens.
    """
    result = (
        "## Style Profile\n\n"
        f"**Color palette:** {profile.color_palette}\n"
        f"**Composition:** {profile.composition}\n"
        f"**Technique:** {profile.technique}\n"
        f"**Mood / atmosphere:** {profile.mood_atmosphere}\n"
        f"**Subject matter:** {profile.subject_matter}\n"
        f"**Influences:** {profile.influences}"
    )
    if not compact:
        result += (
            "\n\n### Gemini raw analysis\n"
            f"{profile.gemini_raw_analysis}\n\n"
            "### Reasoning-model raw analysis\n"
            f"{profile.claude_raw_analysis}"
        )
    return result


def _format_template(template: PromptTemplate) -> str:
    """Render a PromptTemplate into an XML block for the reasoning model to read."""
    parts: list[str] = ["<template>"]
    for section in template.sections:
        parts.append(f'  <section name="{section.name}" description="{section.description}">{section.value}</section>')
    if template.negative_prompt:
        parts.append(f"  <negative>{template.negative_prompt}</negative>")
    if template.caption_sections:
        parts.append(f"  <caption_sections>{', '.join(template.caption_sections)}</caption_sections>")
    if template.caption_length_target > 0:
        parts.append(f"  <caption_length>{template.caption_length_target}</caption_length>")
    parts.append("</template>")
    return "\n".join(parts)


def _format_metrics(metrics: AggregatedMetrics) -> str:
    """Render AggregatedMetrics as a readable summary."""
    d = metrics.summary_dict()
    lines = [f"- {k}: {v:.4f}" for k, v in d.items()]
    return "\n".join(lines)


def _truncate_words(text: str, max_words: int, *, suffix: str = "...") -> str:
    """Cap ``text`` to ``max_words`` whitespace-separated tokens.

    If the text is already short enough, it's returned unchanged; otherwise
    the first ``max_words`` tokens are joined with spaces and ``suffix`` is
    appended.  Used to keep per-image captions and feedback blocks from
    swamping the reasoning-model context.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + suffix

"""Helpers for labeled caption sections and generator prompt shaping."""

from __future__ import annotations

import re

_SECTION_MARKER_RE = re.compile(r"\[(?P<name>[^\]\n]+)\]\s*")
# Floor (word count) below which a caption's [Art Style] block is treated as degenerate and
# replaced with the canon fallback, if one is supplied. Matches the captioner-side floor used
# by :func:`art_style_search.caption._validate_expanded_template`.
_ART_STYLE_MIN_WORDS = 100


def parse_labeled_sections(caption_text: str) -> dict[str, str]:
    """Parse ``[Section]`` blocks from *caption_text*, preserving order."""
    matches = list(_SECTION_MARKER_RE.finditer(caption_text))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        name = match.group("name").strip()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(caption_text)
        sections[name] = caption_text[match.end() : end].strip()
    return sections


def build_generation_prompt(caption_text: str, style_canon: str = "") -> str:
    """Reorder labeled caption blocks so subject leads generation.

    When the caption's ``[Art Style]`` block is missing or below the word floor and a non-empty
    *style_canon* is supplied (typically ``PromptTemplate`` → ``style_foundation.value``), the
    canon is injected as the ``[Art Style]`` block. This guarantees the generator always sees
    the canonical style assertions even when the captioner has misbehaved.
    """
    sections = parse_labeled_sections(caption_text)
    subject = sections.get("Subject", "").strip()
    art_style = sections.get("Art Style", "").strip()

    canon = (style_canon or "").strip()
    if canon and len(art_style.split()) < _ART_STYLE_MIN_WORDS:
        art_style = canon

    if not subject or not art_style:
        return caption_text

    prompt_parts = [
        f"[Subject]\n{subject}",
        f"Render in this style:\n[Art Style]\n{art_style}",
    ]
    for name, body in sections.items():
        if name in {"Subject", "Art Style"} or not body:
            continue
        prompt_parts.append(f"[{name}]\n{body}")
    return "\n\n".join(prompt_parts)

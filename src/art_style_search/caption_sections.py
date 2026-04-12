"""Helpers for labeled caption sections and generator prompt shaping."""

from __future__ import annotations

import re

_SECTION_MARKER_RE = re.compile(r"\[(?P<name>[^\]\n]+)\]\s*")


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


def build_generation_prompt(caption_text: str) -> str:
    """Reorder labeled caption blocks so subject leads generation."""
    sections = parse_labeled_sections(caption_text)
    subject = sections.get("Subject", "").strip()
    art_style = sections.get("Art Style", "").strip()
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

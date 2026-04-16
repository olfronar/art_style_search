"""Response parsing — section regexes, XML-tag extraction, and template validation.

Only the XML-based template parser survives here.  Per-iteration flows consume
reasoning-model JSON via ``prompt.json_contracts`` directly.
"""

from __future__ import annotations

import logging
import re

from art_style_search.types import PromptSection, PromptTemplate
from art_style_search.utils import extract_xml_tag

logger = logging.getLogger(__name__)


# Primary regex: strict name then description order
_SECTION_RE = re.compile(
    r'<section\s+name="(?P<name>[^"]+)"\s+description="(?P<desc>[^"]+)"\s*>'
    r"(?P<value>.*?)"
    r"</section>",
    re.DOTALL,
)
# Fallback regex: allows attributes in any order
_SECTION_RE_LOOSE = re.compile(
    r"<section\s+(?=.*?name=\"(?P<name>[^\"]+)\")(?=.*?description=\"(?P<desc>[^\"]+)\")"
    r"[^>]*>(?P<value>.*?)</section>",
    re.DOTALL,
)


def _parse_template(text: str) -> PromptTemplate:
    """Extract a PromptTemplate from the model's XML-style response.

    Tries strict regex first (name then description order), then falls back
    to a loose regex that accepts attributes in any order.
    """
    sections: list[PromptSection] = []
    for m in _SECTION_RE.finditer(text):
        sections.append(
            PromptSection(
                name=m.group("name").strip(),
                description=m.group("desc").strip(),
                value=m.group("value").strip(),
            )
        )
    # Fallback: try loose regex if strict found nothing
    if not sections:
        for m in _SECTION_RE_LOOSE.finditer(text):
            sections.append(
                PromptSection(
                    name=m.group("name").strip(),
                    description=m.group("desc").strip(),
                    value=m.group("value").strip(),
                )
            )
        if sections:
            logger.warning("Parsed %d sections with loose regex fallback", len(sections))

    neg_raw = extract_xml_tag(text, "negative")
    negative = neg_raw or None

    cs_raw = extract_xml_tag(text, "caption_sections")
    caption_sections = [s.strip() for s in cs_raw.split(",") if s.strip()] if cs_raw else []

    cl_raw = extract_xml_tag(text, "caption_length")
    caption_length_target = int(cl_raw) if cl_raw.isdigit() else 0

    return PromptTemplate(
        sections=sections,
        negative_prompt=negative,
        caption_sections=caption_sections,
        caption_length_target=caption_length_target,
    )


# ---------------------------------------------------------------------------
# Template validation
# ---------------------------------------------------------------------------

# Align validation bounds with the prompt contract for model-produced templates.
# Reasoning prompts still target 2000-8000 words / 8-20 sections; the lowered floor
# below is a safety net that lets laconic experiments (the user's explicit principle)
# pass validation without changing what the reasoner is asked to produce.
_MIN_SECTIONS = 5
_MAX_SECTIONS = 20
_MIN_CAPTION_LENGTH = 500
_MAX_CAPTION_LENGTH = 6000
_MIN_RENDERED_WORDS = 1000
_MAX_RENDERED_WORDS = 8000

# Ordered anchor requirements for sections and caption sections.  Adding a third
# required anchor means one-line edits here — the validator iterates both tables.
_REQUIRED_SECTION_ANCHORS: tuple[tuple[int, str], ...] = (
    (0, "style_foundation"),
    (1, "subject_anchor"),
)
_REQUIRED_CAPTION_ANCHORS: tuple[tuple[int, str], ...] = (
    (0, "Art Style"),
    (1, "Subject"),
)

# Sub-block markers enforced inside the two required anchor sections' `value` text.
# These operationalize the "way of drawing" + "forced proportions" discipline in every
# reasoner-produced template so captions carry the procedural/anatomy spine.
_STYLE_FOUNDATION_DRAWING_MARKER = "how to draw:"
_SUBJECT_ANCHOR_PROPORTION_MARKER = "proportions:"
_SUBJECT_ANCHOR_ARCHETYPE_TOKENS: tuple[str, ...] = (
    "heads tall",
    "heads-tall",
    "chibi",
    "stylized-youth",
    "heroic",
    "realistic-adult",
    "elongated",
)


def _check_anchor_sub_blocks(template: PromptTemplate) -> list[str]:
    """Enforce mandatory sub-blocks inside style_foundation / subject_anchor values."""
    errors: list[str] = []
    by_name = {s.name: s.value for s in template.sections}
    foundation = (by_name.get("style_foundation") or "").lower()
    if foundation and _STYLE_FOUNDATION_DRAWING_MARKER not in foundation:
        errors.append(
            "style_foundation.value must contain a 'How to Draw:' sub-block "
            "(silhouette primitives, construction order, line policy, shading layers, signature quirk)"
        )
    subject = (by_name.get("subject_anchor") or "").lower()
    if subject:
        if _SUBJECT_ANCHOR_PROPORTION_MARKER not in subject:
            errors.append(
                "subject_anchor.value must contain a 'Proportions:' sub-block (heads-tall numeric + archetype)"
            )
        if not any(token in subject for token in _SUBJECT_ANCHOR_ARCHETYPE_TOKENS):
            errors.append(
                "subject_anchor.value must include at least one proportion archetype token: "
                f"{list(_SUBJECT_ANCHOR_ARCHETYPE_TOKENS)}"
            )
    return errors


def _check_anchors(actual: list[str], required: tuple[tuple[int, str], ...], label: str) -> list[str]:
    """Verify each required (position, name) anchor, using ordinal labels in errors."""
    ordinals = ["First", "Second", "Third", "Fourth", "Fifth"]
    errors: list[str] = []
    for index, expected in required:
        position = ordinals[index] if index < len(ordinals) else f"Position-{index + 1}"
        if index < len(actual):
            if actual[index] != expected:
                errors.append(f"{position} {label} must be '{expected}', got '{actual[index]}'")
        else:
            errors.append(f"{position} {label} must be '{expected}', got missing section")
    return errors


def validate_template(
    template: PromptTemplate,
    changed_section: str = "",
    changed_sections: list[str] | None = None,
    risk_level: str = "targeted",
    reference_template: PromptTemplate | None = None,
) -> list[str]:
    """Return a list of validation errors (empty list = valid template).

    Enforces structural invariants that are specified in the prompt text
    but were previously not checked in code.
    """
    errors: list[str] = []

    errors.extend(_check_anchors([s.name for s in template.sections], _REQUIRED_SECTION_ANCHORS, "section"))
    errors.extend(_check_anchors(list(template.caption_sections), _REQUIRED_CAPTION_ANCHORS, "caption section"))
    errors.extend(_check_anchor_sub_blocks(template))

    n = len(template.sections)
    if n < _MIN_SECTIONS or n > _MAX_SECTIONS:
        errors.append(f"Section count {n} outside bounds [{_MIN_SECTIONS}, {_MAX_SECTIONS}]")

    clt = template.caption_length_target
    if clt != 0 and (clt < _MIN_CAPTION_LENGTH or clt > _MAX_CAPTION_LENGTH):
        errors.append(f"Caption length target {clt} outside bounds [{_MIN_CAPTION_LENGTH}, {_MAX_CAPTION_LENGTH}]")

    rendered_words = len(template.render().split())
    if rendered_words < _MIN_RENDERED_WORDS or rendered_words > _MAX_RENDERED_WORDS:
        errors.append(
            f"Rendered prompt word count {rendered_words} outside bounds [{_MIN_RENDERED_WORDS}, {_MAX_RENDERED_WORDS}]"
        )

    normalized_changed_sections = list(changed_sections or [])
    if not normalized_changed_sections and changed_section:
        normalized_changed_sections = [changed_section]

    names = {s.name for s in template.sections}
    reference_names = {s.name for s in reference_template.sections} if reference_template is not None else set()
    structural_names = {"caption_sections", "caption_length_target", "negative_prompt"}
    allowed_changed_names = names | reference_names | structural_names
    if changed_section and changed_section not in allowed_changed_names:
        errors.append(f"changed_section '{changed_section}' not in template sections: {sorted(names)}")
    for section_name in normalized_changed_sections:
        if section_name == changed_section:
            continue
        if section_name not in allowed_changed_names:
            errors.append(f"changed_sections contains '{section_name}' not in template sections: {sorted(names)}")

    if risk_level == "targeted" and len(normalized_changed_sections) > 1:
        errors.append("targeted experiments must change exactly 1 section")
    if risk_level == "bold" and len(normalized_changed_sections) > 3:
        errors.append("bold experiments may change up to 3 related sections")

    return errors

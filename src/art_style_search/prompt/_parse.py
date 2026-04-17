"""Template validation shared by the JSON payload validators.

Reasoning-model exchanges travel as JSON via ``prompt.json_contracts``; this
module now carries only the structural invariants that validate a parsed
``PromptTemplate`` (required anchors, section-count bounds, rendered-word
bounds, and changed-section rules per risk level).
"""

from __future__ import annotations

import re

from art_style_search.types import PromptTemplate

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

# Style canon anti-methodology lint. The canon (``style_foundation.value``) must hold concrete
# assertive style content, not instructions to the captioner. These patterns catch the drift
# mode we observed in prior runs where the reasoner wrapped canon content in imperative/audit
# scaffolding (slot numbers, checkboxes, "Write the block as…", "Target N-M words", etc.).
# Each pattern is line-anchored where practical; a single hit is grounds to reject.
_CANON_METHODOLOGY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*slot\s+\d+\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*-\s*\[\s*\]", re.MULTILINE),
    re.compile(r"\bwrite\s+the\s+(?:\[art\s+style\]|block|canon)\b", re.IGNORECASE),
    re.compile(r"\btarget\s+\d+\s*[-\u2013]\s*\d+\s+words?\b", re.IGNORECASE),
    re.compile(r"\bbegin\s+the\s+block\b", re.IGNORECASE),
    re.compile(r"^\s*mandatory\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bdeclare\s+the\s+medium\b", re.IGNORECASE),
)


def _check_anchor_sub_blocks(template: PromptTemplate) -> list[str]:
    """Enforce mandatory sub-blocks inside style_foundation / subject_anchor values."""
    errors: list[str] = []
    by_name = {s.name: s.value for s in template.sections}
    foundation_raw = by_name.get("style_foundation") or ""
    foundation = foundation_raw.lower()
    if foundation and _STYLE_FOUNDATION_DRAWING_MARKER not in foundation:
        errors.append(
            "style_foundation.value must contain a 'How to Draw:' sub-block "
            "(silhouette primitives, construction order, line policy, shading layers, signature quirk)"
        )
    if foundation_raw:
        methodology_hits = [p.pattern for p in _CANON_METHODOLOGY_PATTERNS if p.search(foundation_raw)]
        if methodology_hits:
            errors.append(
                "style_foundation.value reads as captioner methodology, not style canon — drop "
                "imperative/audit markers and assert the style directly "
                f"(matched {len(methodology_hits)} methodology patterns: {methodology_hits[:3]})"
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

"""Response parsing — regexes, XML-tag extraction, and the RefinementResult dataclass.

These helpers consume Claude / GLM / GPT response text and turn it into typed data.
All parsing lives here so the higher-level flow modules (``experiments``, ``synthesis``,
``review``) can focus on prompt construction and API calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

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
_CONVERGED_RE = re.compile(r"\[CONVERGED\]")
_BRANCH_BLOCK_RE = re.compile(r"<branch\b[^>]*>(.*?)</branch>", re.DOTALL)


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


def _parse_analysis(text: str) -> str:
    return extract_xml_tag(text, "analysis")


def _parse_template_changes(text: str) -> str:
    return extract_xml_tag(text, "template_changes")


def _parse_converged(text: str) -> bool:
    return bool(_CONVERGED_RE.search(text))


def _parse_hypothesis(text: str) -> str:
    return extract_xml_tag(text, "hypothesis")


def _parse_experiment(text: str) -> str:
    return extract_xml_tag(text, "experiment")


@dataclass
class Lessons:
    """Structured lessons from one iteration."""

    confirmed: str = ""
    rejected: str = ""
    new_insight: str = ""


@dataclass
class RefinementResult:
    """Complete result of a template refinement by the reasoning model."""

    template: PromptTemplate
    analysis: str
    template_changes: str
    should_stop: bool
    hypothesis: str
    experiment: str
    lessons: Lessons
    builds_on: str | None
    open_problems: list[str]
    changed_section: str = ""
    target_category: str = ""


def _parse_lessons(text: str) -> Lessons:
    return Lessons(
        confirmed=extract_xml_tag(text, "confirmed"),
        rejected=extract_xml_tag(text, "rejected"),
        new_insight=extract_xml_tag(text, "new_insight"),
    )


def _parse_changed_section(text: str) -> str:
    """Extract the <changed_section> tag — returns section name or empty string."""
    return extract_xml_tag(text, "changed_section")


def _parse_target_category(text: str) -> str:
    """Extract the <target_category> tag — returns category name or empty string."""
    return extract_xml_tag(text, "target_category")


def _parse_builds_on(text: str) -> str | None:
    """Extract the <builds_on> tag — returns hypothesis IDs or None."""
    val = extract_xml_tag(text, "builds_on")
    if not val:
        return None
    return val if val.lower() != "none" else None


def _parse_open_problems(text: str) -> list[str]:
    """Extract numbered open problems from <open_problems> tag."""
    raw = extract_xml_tag(text, "open_problems")
    if not raw:
        return []
    # Split on numbered lines: "1. ...", "2. ..." etc.
    items = re.split(r"\n\s*\d+\.\s+", "\n" + raw)
    return [item.strip() for item in items if item.strip()]


def _parse_initial_templates(text: str, num_branches: int) -> list[PromptTemplate]:
    """Parse multiple templates from the initial proposal response.

    The reasoning model wraps each template in a <branch> tag.  Fall back to
    parsing a single template and duplicating it if the expected structure is
    absent.
    """
    blocks = _BRANCH_BLOCK_RE.findall(text)
    # Fallback: parse the whole response as a single template if no <branch> tags found
    templates = [_parse_template(block) for block in blocks] if blocks else [_parse_template(text)]

    # Pad if the model produced fewer than requested
    while len(templates) < num_branches:
        templates.append(templates[-1])

    return templates[:num_branches]


def _parse_refinement_branches(text: str, num_experiments: int) -> list[RefinementResult]:
    """Parse multiple RefinementResults from <branch>-wrapped response blocks.

    Each branch is expected to contain the same response tags as a single
    refinement call (hypothesis, experiment, template, etc.).
    """
    blocks = _BRANCH_BLOCK_RE.findall(text)
    if not blocks:
        # Fallback: treat entire response as one branch
        logger.warning("No <branch> tags found — treating response as a single experiment")
        blocks = [text]

    results: list[RefinementResult] = []
    for i, block in enumerate(blocks):
        new_template = _parse_template(block)
        if not new_template.sections:
            logger.warning("Branch %d has no sections — skipping", i)
            continue

        results.append(
            RefinementResult(
                template=new_template,
                analysis=_parse_analysis(block),
                template_changes=_parse_template_changes(block),
                should_stop=_parse_converged(block),
                hypothesis=_parse_hypothesis(block),
                experiment=_parse_experiment(block),
                lessons=_parse_lessons(block),
                builds_on=_parse_builds_on(block),
                open_problems=_parse_open_problems(block),
                changed_section=_parse_changed_section(block),
                target_category=_parse_target_category(block),
            )
        )

    if len(results) < num_experiments:
        logger.warning("Got %d valid branches but requested %d", len(results), num_experiments)

    return results

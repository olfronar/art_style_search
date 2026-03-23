"""Claude API calls for meta-prompt optimization.

Claude operates on both the prompt template (structure/sections) and the prompt
values (content within sections).  It proposes initial diverse templates and
iteratively refines them based on evaluation metrics.
"""

from __future__ import annotations

import logging
import re

import anthropic

from art_style_search.types import (
    AggregatedMetrics,
    BranchState,
    PromptSection,
    PromptTemplate,
    StyleProfile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_style_profile(profile: StyleProfile) -> str:
    """Render a StyleProfile into a text block for system prompts."""
    return (
        "## Style Profile\n\n"
        f"**Color palette:** {profile.color_palette}\n"
        f"**Composition:** {profile.composition}\n"
        f"**Technique:** {profile.technique}\n"
        f"**Mood / atmosphere:** {profile.mood_atmosphere}\n"
        f"**Subject matter:** {profile.subject_matter}\n"
        f"**Influences:** {profile.influences}\n\n"
        "### Gemini raw analysis\n"
        f"{profile.gemini_raw_analysis}\n\n"
        "### Claude raw analysis\n"
        f"{profile.claude_raw_analysis}"
    )


def _format_template(template: PromptTemplate) -> str:
    """Render a PromptTemplate into an XML block for Claude to read."""
    parts: list[str] = ["<template>"]
    for section in template.sections:
        parts.append(f'  <section name="{section.name}" description="{section.description}">{section.value}</section>')
    if template.negative_prompt:
        parts.append(f"  <negative>{template.negative_prompt}</negative>")
    parts.append("</template>")
    return "\n".join(parts)


def _format_metrics(metrics: AggregatedMetrics) -> str:
    """Render AggregatedMetrics as a readable summary."""
    d = metrics.summary_dict()
    lines = [f"- {k}: {v:.4f}" for k, v in d.items()]
    return "\n".join(lines)


def _format_history_tail(branch: BranchState, max_entries: int = 5) -> str:
    """Format the last *max_entries* iterations of a branch for context."""
    recent = branch.history[-max_entries:]
    if not recent:
        return "(no history yet)"
    parts: list[str] = []
    for r in recent:
        kept_tag = "KEPT" if r.kept else "DISCARDED"
        parts.append(
            f"### Iteration {r.iteration} [{kept_tag}]\n"
            f"Rendered prompt: {r.rendered_prompt}\n"
            f"Metrics:\n{_format_metrics(r.aggregated)}\n"
            f"Analysis: {r.claude_analysis}\n"
            f"Template changes: {r.template_changes}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r'<section\s+name="(?P<name>[^"]+)"\s+description="(?P<desc>[^"]+)"\s*>'
    r"(?P<value>.*?)"
    r"</section>",
    re.DOTALL,
)
_NEGATIVE_RE = re.compile(r"<negative>(.*?)</negative>", re.DOTALL)
_ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL)
_TEMPLATE_CHANGES_RE = re.compile(r"<template_changes>(.*?)</template_changes>", re.DOTALL)
_CONVERGED_RE = re.compile(r"\[CONVERGED\]")


def _parse_template(text: str) -> PromptTemplate:
    """Extract a PromptTemplate from Claude's XML-style response."""
    sections: list[PromptSection] = []
    for m in _SECTION_RE.finditer(text):
        sections.append(
            PromptSection(
                name=m.group("name").strip(),
                description=m.group("desc").strip(),
                value=m.group("value").strip(),
            )
        )
    negative = None
    neg_match = _NEGATIVE_RE.search(text)
    if neg_match:
        negative = neg_match.group(1).strip() or None
    return PromptTemplate(sections=sections, negative_prompt=negative)


def _parse_analysis(text: str) -> str:
    m = _ANALYSIS_RE.search(text)
    return m.group(1).strip() if m else ""


def _parse_template_changes(text: str) -> str:
    m = _TEMPLATE_CHANGES_RE.search(text)
    return m.group(1).strip() if m else ""


def _parse_converged(text: str) -> bool:
    return bool(_CONVERGED_RE.search(text))


# ---------------------------------------------------------------------------
# Multi-template initial response parsing
# ---------------------------------------------------------------------------

_BRANCH_BLOCK_RE = re.compile(r"<branch\b[^>]*>(.*?)</branch>", re.DOTALL)


def _parse_initial_templates(text: str, num_branches: int) -> list[PromptTemplate]:
    """Parse multiple templates from the initial proposal response.

    Claude wraps each template in a <branch> tag.  Fall back to parsing a
    single template and duplicating it if the expected structure is absent.
    """
    blocks = _BRANCH_BLOCK_RE.findall(text)
    # Fallback: parse the whole response as a single template if no <branch> tags found
    templates = [_parse_template(block) for block in blocks] if blocks else [_parse_template(text)]

    # Pad if Claude produced fewer than requested
    while len(templates) < num_branches:
        templates.append(templates[-1])

    return templates[:num_branches]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def propose_initial_templates(
    style_profile: StyleProfile,
    num_branches: int,
    *,
    client: anthropic.AsyncAnthropic,
    model: str,
) -> list[PromptTemplate]:
    """Generate diverse initial prompt templates for population branches."""

    system = (
        "You are an expert art director and prompt engineer specializing in "
        "text-to-image generation.  Your task is to create diverse prompt "
        "templates that capture a specific art style.\n\n"
        "Each template consists of named sections, each with a description of "
        "its purpose and a value containing the actual prompt text.  Different "
        "templates should explore different structural approaches:\n"
        "- Vary the NUMBER of sections (some concise, some detailed).\n"
        "- Vary the FOCUS (some emphasize technique, others mood, others color).\n"
        "- Vary the PHRASING style (descriptive prose vs. keyword lists vs. "
        "art-director instructions).\n\n"
        "You may also include a <negative> tag with things to avoid.\n\n"
        f"Produce exactly {num_branches} templates, each wrapped in a <branch> "
        "tag.\n\n"
        "Response format (repeat for each branch):\n"
        "<branch>\n"
        "<template>\n"
        '  <section name="..." description="...">value</section>\n'
        "  ...\n"
        "  <negative>things to avoid</negative>\n"
        "</template>\n"
        "</branch>"
    )

    user = (
        "Based on the following style profile, create the initial prompt "
        "templates.\n\n"
        f"{_format_style_profile(style_profile)}"
    )

    logger.info("Requesting %d initial templates from Claude (%s)", num_branches, model)

    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = response.content[0].text

    templates = _parse_initial_templates(text, num_branches)

    for i, t in enumerate(templates):
        if not t.sections:
            logger.warning("Branch %d initial template has no sections — raw response may need review", i)

    return templates


async def refine_template(
    style_profile: StyleProfile,
    branch: BranchState,
    global_best: PromptTemplate | None,
    global_best_metrics: AggregatedMetrics | None,
    *,
    client: anthropic.AsyncAnthropic,
    model: str,
) -> tuple[PromptTemplate, str, str, bool]:
    """Refine a branch's template based on metric feedback.

    Returns
    -------
    new_template : PromptTemplate
        The proposed (possibly restructured) template.
    analysis_text : str
        Claude's reasoning about changes.
    template_changes_description : str
        Short description of structural changes, if any.
    should_stop : bool
        True if Claude judges the branch has converged.
    """

    system = (
        "You are an expert art director and prompt engineer iterating on a "
        "text-to-image prompt to reproduce a specific art style.\n\n"
        "You can change BOTH the template structure (add/remove/rename "
        "sections) AND the values within sections.  Make targeted, "
        "evidence-based edits: look at which metrics improved or regressed "
        "and adjust accordingly.\n\n"
        "Metric guidance:\n"
        "- DINO similarity (higher=better): semantic/structural match to "
        "references.\n"
        "- LPIPS distance (lower=better): perceptual distance.\n"
        "- HPS v2 (higher=better): human preference for text-image "
        "alignment.\n"
        "- LAION Aesthetics (higher=better, 1-10): overall aesthetic quality.\n\n"
        "If the metrics have plateaued and further improvement seems unlikely, "
        "append [CONVERGED] at the very end of your response.\n\n"
        "Response format:\n"
        "<template_changes>describe structural changes, or 'none'</template_changes>\n"
        "<template>\n"
        '  <section name="..." description="...">value</section>\n'
        "  ...\n"
        "  <negative>things to avoid</negative>\n"
        "</template>\n"
        "<analysis>your reasoning</analysis>\n"
        "[CONVERGED]  (only if converged)"
    )

    # Build the user message with all context
    user_parts: list[str] = [
        "## Style Profile\n",
        _format_style_profile(style_profile),
        "\n\n## Current Branch Template\n",
        _format_template(branch.current_template),
        f"\nRendered prompt: {branch.current_template.render()}",
    ]

    if branch.best_metrics:
        user_parts.append("\n\n## Branch Best Metrics\n")
        user_parts.append(_format_metrics(branch.best_metrics))

    user_parts.append("\n\n## Recent Iteration History\n")
    user_parts.append(_format_history_tail(branch))

    if global_best is not None and global_best_metrics is not None:
        user_parts.append("\n\n## Global Best (across all branches — for cross-pollination)\n")
        user_parts.append(_format_template(global_best))
        user_parts.append(f"\nRendered prompt: {global_best.render()}")
        user_parts.append(f"\nMetrics:\n{_format_metrics(global_best_metrics)}")

    user_parts.append("\n\nPropose an improved template.  Focus on the weakest metrics while maintaining strengths.")

    user = "".join(user_parts)

    logger.info("Requesting template refinement for branch %d from Claude (%s)", branch.branch_id, model)

    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = response.content[0].text

    new_template = _parse_template(text)
    analysis_text = _parse_analysis(text)
    template_changes_description = _parse_template_changes(text)
    should_stop = _parse_converged(text)

    if not new_template.sections:
        logger.warning(
            "Refined template for branch %d has no sections — falling back to current template",
            branch.branch_id,
        )
        new_template = branch.current_template

    return new_template, analysis_text, template_changes_description, should_stop

"""Claude API calls for meta-prompt optimization.

Claude operates on both the prompt template (structure/sections) and the prompt
values (content within sections).  It proposes initial diverse templates and
iteratively refines them based on evaluation metrics.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import anthropic

from art_style_search.types import (
    AggregatedMetrics,
    BranchState,
    PromptSection,
    PromptTemplate,
    StyleProfile,
)
from art_style_search.utils import extract_text, stream_message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


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
            "### Claude raw analysis\n"
            f"{profile.claude_raw_analysis}"
        )
    return result


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
_HYPOTHESIS_RE = re.compile(r"<hypothesis>(.*?)</hypothesis>", re.DOTALL)
_EXPERIMENT_RE = re.compile(r"<experiment>(.*?)</experiment>", re.DOTALL)
_CONFIRMED_RE = re.compile(r"<confirmed>(.*?)</confirmed>", re.DOTALL)
_REJECTED_RE = re.compile(r"<rejected>(.*?)</rejected>", re.DOTALL)
_NEW_INSIGHT_RE = re.compile(r"<new_insight>(.*?)</new_insight>", re.DOTALL)


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


def _parse_hypothesis(text: str) -> str:
    m = _HYPOTHESIS_RE.search(text)
    return m.group(1).strip() if m else ""


def _parse_experiment(text: str) -> str:
    m = _EXPERIMENT_RE.search(text)
    return m.group(1).strip() if m else ""


@dataclass
class Lessons:
    """Structured lessons from one iteration."""

    confirmed: str = ""
    rejected: str = ""
    new_insight: str = ""


def _parse_lessons(text: str) -> Lessons:
    confirmed = _CONFIRMED_RE.search(text)
    rejected = _REJECTED_RE.search(text)
    insight = _NEW_INSIGHT_RE.search(text)
    return Lessons(
        confirmed=confirmed.group(1).strip() if confirmed else "",
        rejected=rejected.group(1).strip() if rejected else "",
        new_insight=insight.group(1).strip() if insight else "",
    )


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
        "You are an expert art director and prompt engineer. Your task is to create "
        "diverse META-PROMPTS — instructions that tell an AI vision model (Gemini Pro) "
        "HOW to describe/caption reference images so that the captions can be used by "
        "an image generation model (Gemini Flash) to recreate the originals as precisely as possible.\n\n"
        "## How the system works\n"
        "meta-prompt + reference image → Gemini Pro caption → Gemini Flash generation → compare with original.\n"
        "The meta-prompt is the ONLY thing being optimized. It must instruct the captioner "
        "to describe every detail needed for faithful recreation.\n\n"
        "## Meta-prompt requirements\n"
        "- 6-10 sections, each instructing the captioner WHAT to describe and HOW precisely.\n"
        "- Must cover: technique/medium, colors, composition, characters/figures, "
        "background/environment, textures/details, lighting, mood/atmosphere.\n"
        "- Each section should be 2-4 sentences of instruction.\n"
        "- Total rendered meta-prompt should be 200-400 words.\n"
        "- Include a negative section: what the captioner should tell the generator to AVOID.\n"
        "- The meta-prompt must produce captions specific enough that someone who has never "
        "seen the image could recreate it from the caption alone.\n\n"
        "## Example of a good meta-prompt section\n"
        '<section name="colors_and_palette" description="instruct captioner on color description">'
        "Describe the EXACT colors visible in the image using specific color names "
        "(e.g. 'burnt sienna', 'cadmium yellow', not just 'brown' or 'yellow'). "
        "Note the overall color temperature (warm/cool), saturation levels, "
        "and how colors relate to each other. Describe any gradients or color transitions."
        "</section>\n\n"
        "## Diversity across meta-prompts\n"
        "- Vary emphasis: some focus on technique precision, others on spatial accuracy, "
        "others on mood fidelity.\n"
        "- Vary instruction style: some give the captioner strict checklists, "
        "others give artistic direction, others ask for technical analysis.\n"
        "- All must be comprehensive — diversity is in approach, not coverage.\n\n"
        f"Produce exactly {num_branches} meta-prompts, each wrapped in a <branch> tag.\n\n"
        "Response format (repeat for each branch):\n"
        "<branch>\n"
        "<template>\n"
        '  <section name="..." description="what this instructs the captioner to describe">'
        "instruction for the captioner (2-4 sentences)</section>\n"
        "  ... (6-10 sections)\n"
        "  <negative>instruct captioner to tell generator what to avoid</negative>\n"
        "</template>\n"
        "</branch>"
    )

    user = (
        "Based on the following style profile of the reference images, create the initial "
        "meta-prompts. Remember: these are INSTRUCTIONS for a captioner, not direct image prompts.\n\n"
        f"{_format_style_profile(style_profile)}"
    )

    logger.info("Requesting %d initial templates from Claude (%s)", num_branches, model)

    response = await stream_message(
        client,
        model=model,
        max_tokens=80000,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = extract_text(response)

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
    vision_feedback: str = "",
    roundtrip_feedback: str = "",
    caption_diffs: str = "",
) -> tuple[PromptTemplate, str, str, bool, str, str, Lessons]:
    """Refine a branch's template based on metric feedback.

    Returns (new_template, analysis, template_changes, should_stop,
             hypothesis, experiment, lessons).
    """

    system = (
        "You are an expert art director and prompt engineer optimizing a META-PROMPT.\n\n"
        "## How this system works\n"
        "You are NOT writing image generation prompts directly. You are writing a META-PROMPT — "
        "instructions that tell Gemini Pro HOW to caption/describe reference images. "
        "Those captions are then used by Gemini Flash to generate images. "
        "The goal is to produce captions precise enough to RECREATE the original images.\n\n"
        "The pipeline: meta-prompt + reference image → Gemini Pro caption → Gemini Flash generation → compare with original.\n\n"
        "## What makes a good meta-prompt\n"
        "The meta-prompt must instruct the captioner to describe EVERY visual detail needed "
        "for faithful recreation:\n"
        "- Exact colors, technique, medium, brushwork\n"
        "- Character/figure details: poses, expressions, clothing, proportions\n"
        "- Background/environment: setting, architecture, nature elements\n"
        "- Composition: layout, framing, depth, perspective\n"
        "- Lighting, shadows, atmospheric effects\n"
        "- Textures, patterns, fine details\n"
        "- Mood, emotional tone\n"
        "- What to AVOID (common AI generation artifacts)\n\n"
        "The meta-prompt should be 6-10 sections, each instructing the captioner what to describe "
        "and how detailed to be. Total rendered prompt should be 200-400 words.\n\n"
        "## Metric guidance (with typical ranges)\n"
        "These compare generated images against the SPECIFIC originals they were captioned from:\n"
        "- DINO similarity (higher=better): semantic/structural match. "
        "0.2=unrelated, 0.4=somewhat similar, 0.6=good match, 0.8+=very close.\n"
        "- LPIPS distance (lower=better): perceptual distance. "
        "0.7=very different, 0.5=noticeable differences, 0.3=similar, 0.1=near identical.\n"
        "- HPS v2 (higher=better): human preference for prompt-image alignment. "
        "Typical range 0.20-0.30.\n"
        "- LAION Aesthetics (higher=better, 1-10): aesthetic quality. "
        "5=mediocre, 7=good, 8+=excellent.\n\n"
        "## Iteration strategy\n"
        "- Make 1-2 targeted changes per iteration, not wholesale rewrites.\n"
        "- If DINO/LPIPS are weak: the captions miss structural or color details — "
        "add instructions for the captioner to be more specific about those.\n"
        "- If per-image scores vary widely: some images are harder — add instructions "
        "for handling complex scenes, multiple subjects, or unusual compositions.\n"
        "- Use the vision comparison and per-image roundtrip feedback to identify "
        "what the captions consistently miss.\n"
        "- CRITICAL: Read the Research Log carefully. Do NOT repeat experiments that were "
        "already rejected. Build on confirmed insights. Reference specific log entries.\n\n"
        "If metrics have plateaued, append [CONVERGED] at the very end.\n\n"
        "Response format (ALL tags required):\n"
        "<lessons>\n"
        "  <confirmed>Which previous hypotheses are confirmed by THIS iteration's results?</confirmed>\n"
        "  <rejected>Which previous hypotheses are rejected? What didn't work and why?</rejected>\n"
        "  <new_insight>Any new observation from the data not covered by existing hypotheses</new_insight>\n"
        "</lessons>\n"
        "<hypothesis>Based on the research log and current results, what do you believe "
        "is the PRIMARY remaining gap? Be specific — name the metric, the images, the visual element.</hypothesis>\n"
        "<experiment>The specific 1-2 changes you're making to test this hypothesis</experiment>\n"
        "<template_changes>structural changes or 'none'</template_changes>\n"
        "<template>\n"
        '  <section name="..." description="...">value</section>\n'
        "  ...\n"
        "  <negative>things to avoid</negative>\n"
        "</template>\n"
        "[CONVERGED]  (only if converged)"
    )

    # Build the user message with all context
    # Use compact profile (no raw analyses) after first iteration to save tokens
    has_history = len(branch.history) > 0
    user_parts: list[str] = [
        "## Style Profile\n",
        _format_style_profile(style_profile, compact=has_history),
        "\n\n## Current Branch Template\n",
        _format_template(branch.current_template),
        f"\nRendered prompt: {branch.current_template.render()}",
    ]

    if branch.best_metrics:
        user_parts.append("\n\n## Branch Best Metrics\n")
        user_parts.append(_format_metrics(branch.best_metrics))

    # Research log — accumulated lessons from all previous iterations
    if branch.research_log:
        user_parts.append("\n\n## Research Log (accumulated lessons — DO NOT repeat rejected experiments)\n")
        user_parts.append(branch.research_log)

    # Only show last iteration's details (not full history)
    if branch.history:
        last = branch.history[-1]
        user_parts.append(f"\n\n## Last Iteration ({last.iteration}) [{('KEPT' if last.kept else 'DISCARDED')}]\n")
        user_parts.append(f"Metrics:\n{_format_metrics(last.aggregated)}\n")
        if last.hypothesis:
            user_parts.append(f"Hypothesis tested: {last.hypothesis}\n")
        if last.experiment:
            user_parts.append(f"Experiment: {last.experiment}\n")

    if global_best is not None and global_best_metrics is not None:
        user_parts.append("\n\n## Global Best (across all branches — for cross-pollination)\n")
        user_parts.append(_format_template(global_best))
        user_parts.append(f"\nRendered prompt: {global_best.render()}")
        user_parts.append(f"\nMetrics:\n{_format_metrics(global_best_metrics)}")

    if vision_feedback:
        user_parts.append("\n\n## Vision Comparison (Gemini analysis of generated vs reference images)\n")
        user_parts.append(vision_feedback)

    if roundtrip_feedback:
        user_parts.append("\n\n## Per-Image Results (sorted worst → best by DINO)\n")
        user_parts.append(roundtrip_feedback)

    if caption_diffs:
        user_parts.append(f"\n\n{caption_diffs}")

    has_feedback = vision_feedback or roundtrip_feedback
    instruction = (
        "\n\nPropose an improved template. Review the Research Log, then formulate a hypothesis and experiment."
    )
    if has_feedback:
        instruction += (
            " Use the vision comparison and per-image results to ground your hypothesis in specific evidence."
        )
    user_parts.append(instruction)

    user = "".join(user_parts)

    logger.info("Requesting template refinement for branch %d from Claude (%s)", branch.branch_id, model)

    response = await stream_message(
        client,
        model=model,
        max_tokens=80000,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = extract_text(response)

    new_template = _parse_template(text)
    analysis_text = _parse_analysis(text)
    template_changes_description = _parse_template_changes(text)
    should_stop = _parse_converged(text)
    hypothesis = _parse_hypothesis(text)
    experiment = _parse_experiment(text)
    lessons = _parse_lessons(text)

    if not new_template.sections:
        logger.warning(
            "Refined template for branch %d has no sections — falling back to current template",
            branch.branch_id,
        )
        new_template = branch.current_template

    return new_template, analysis_text, template_changes_description, should_stop, hypothesis, experiment, lessons

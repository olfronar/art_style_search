"""Claude API calls for meta-prompt optimization.

Claude operates on both the prompt template (structure/sections) and the prompt
values (content within sections).  It proposes initial diverse templates and
iteratively refines them based on evaluation metrics.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from art_style_search.types import (
    AggregatedMetrics,
    IterationResult,
    KnowledgeBase,
    PromptSection,
    PromptTemplate,
    StyleProfile,
)
from art_style_search.utils import ReasoningClient

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
_BUILDS_ON_RE = re.compile(r"<builds_on>(.*?)</builds_on>", re.DOTALL)
_OPEN_PROBLEMS_RE = re.compile(r"<open_problems>(.*?)</open_problems>", re.DOTALL)


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


def _parse_builds_on(text: str) -> str | None:
    """Extract the <builds_on> tag — returns hypothesis IDs or None."""
    m = _BUILDS_ON_RE.search(text)
    if not m:
        return None
    val = m.group(1).strip()
    return val if val.lower() != "none" else None


def _parse_open_problems(text: str) -> list[str]:
    """Extract numbered open problems from <open_problems> tag."""
    m = _OPEN_PROBLEMS_RE.search(text)
    if not m:
        return []
    raw = m.group(1).strip()
    # Split on numbered lines: "1. ...", "2. ..." etc.
    items = re.split(r"\n\s*\d+\.\s+", "\n" + raw)
    return [item.strip() for item in items if item.strip()]


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
    client: ReasoningClient,
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

    logger.info("Requesting %d initial templates (%s)", num_branches, model)

    text = await client.call(model=model, system=system, user=user, max_tokens=80000)

    templates = _parse_initial_templates(text, num_branches)

    for i, t in enumerate(templates):
        if not t.sections:
            logger.warning("Branch %d initial template has no sections — raw response may need review", i)

    return templates


async def refine_template(
    style_profile: StyleProfile,
    current_template: PromptTemplate,
    knowledge_base: KnowledgeBase,
    best_metrics: AggregatedMetrics | None,
    last_results: list[IterationResult] | None,
    *,
    client: ReasoningClient,
    model: str,
    vision_feedback: str = "",
    roundtrip_feedback: str = "",
    caption_diffs: str = "",
    already_proposed: list[str] | None = None,
) -> tuple[PromptTemplate, str, str, bool, str, str, Lessons, str | None, list[str]]:
    """Propose a template refinement based on shared Knowledge Base.

    No persistent branch identity — each call proposes one experiment.
    Use *already_proposed* to prevent duplicate hypotheses within an iteration.

    Returns (new_template, analysis, template_changes, should_stop,
             hypothesis, experiment, lessons, builds_on, open_problems).
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
        "- Character/figure details: poses, expressions, clothing, proportions, identity\n"
        "- Background/environment: setting, architecture, nature elements\n"
        "- Composition: layout, framing, depth, perspective\n"
        "- Lighting, shadows, atmospheric effects\n"
        "- Textures, patterns, fine details\n"
        "- Mood, emotional tone\n"
        "- What to AVOID (common AI generation artifacts)\n\n"
        "The meta-prompt should be 6-10 sections, each instructing the captioner what to describe "
        "and how detailed to be. Total rendered prompt should be 200-400 words.\n\n"
        "## Metric guidance (with typical ranges)\n"
        "Metrics compare each generated image against its SPECIFIC paired original (not all references):\n"
        "- DINO similarity (higher=better): semantic/structural match per image pair. "
        "0.2=unrelated, 0.4=somewhat similar, 0.6=good match, 0.8+=very close.\n"
        "- LPIPS distance (lower=better): perceptual distance per image pair. "
        "0.7=very different, 0.5=noticeable differences, 0.3=similar, 0.1=near identical.\n"
        "- HPS v2 (higher=better): how well the generated image matches the caption. "
        "Typical range 0.20-0.30.\n"
        "- LAION Aesthetics (higher=better, 1-10): aesthetic quality. "
        "5=mediocre, 7=good, 8+=excellent.\n\n"
        "## Iteration strategy\n"
        "- You are proposing ONE of several parallel experiments this iteration. "
        "Each experiment tests a different hypothesis.\n"
        "- There are no fixed 'branches' — shift focus freely between categories "
        "as the weakest area changes.\n"
        "- Make 1-2 targeted changes per iteration, not wholesale rewrites.\n"
        "- If DINO/LPIPS are weak: the captions miss structural or color details — "
        "add instructions for the captioner to be more specific about those.\n"
        "- If per-image scores vary widely: some images are harder — consider "
        "conditional captioning instructions (e.g. 'for character images describe X; "
        "for backgrounds describe Y').\n"
        "- Use the vision comparison and per-image roundtrip feedback to identify "
        "what the captions consistently miss.\n"
        "- CRITICAL: Read the Knowledge Base carefully. The 'Do NOT Repeat' section lists "
        "failed experiments. Build on confirmed insights. Reference hypothesis IDs (e.g. 'builds on H3').\n"
        "- Use Per-Category Status to identify which style dimensions need work.\n"
        "- Target the weakest category or build on partially confirmed hypotheses.\n"
        "- Use the Open Problems list to focus on the highest-priority gaps.\n"
        "- Update <open_problems> each iteration: add new ones, remove solved ones, re-rank.\n\n"
        "If metrics have plateaued, append [CONVERGED] at the very end.\n\n"
        "Response format (ALL tags required):\n"
        "<lessons>\n"
        "  <confirmed>Which previous hypotheses are confirmed by THIS iteration's results?</confirmed>\n"
        "  <rejected>Which previous hypotheses are rejected? What didn't work and why?</rejected>\n"
        "  <new_insight>Any new observation from the data not covered by existing hypotheses</new_insight>\n"
        "</lessons>\n"
        "<hypothesis>Based on the knowledge base and current results, what is the "
        "PRIMARY remaining gap? Be specific — name the metric, the images, the visual element.</hypothesis>\n"
        "<builds_on>H-ids this builds on, or 'none' for fresh direction</builds_on>\n"
        "<experiment>The specific 1-2 changes you're making to test this hypothesis</experiment>\n"
        "<open_problems>\n"
        "  1. Most critical remaining problem\n"
        "  2. Second most critical\n"
        "  3. Third (if any)\n"
        "</open_problems>\n"
        "<template_changes>structural changes or 'none'</template_changes>\n"
        "<template>\n"
        '  <section name="..." description="...">value</section>\n'
        "  ...\n"
        "  <negative>things to avoid</negative>\n"
        "</template>\n"
        "[CONVERGED]  (only if converged)"
    )

    # Build the user message with all context
    has_history = knowledge_base.hypotheses
    user_parts: list[str] = [
        "## Style Profile\n",
        _format_style_profile(style_profile, compact=bool(has_history)),
        "\n\n## Current Template\n",
        _format_template(current_template),
        f"\nRendered prompt: {current_template.render()}",
    ]

    if best_metrics:
        user_parts.append("\n\n## Best Metrics So Far\n")
        user_parts.append(_format_metrics(best_metrics))

    # Knowledge base — structured lessons from all previous experiments
    kb_text = knowledge_base.render_for_claude()
    if kb_text:
        user_parts.append("\n\n")
        user_parts.append(kb_text)

    # Show last iteration results — only the kept experiment in detail, others as one-liners
    if last_results:
        user_parts.append("\n\n## Last Iteration Results\n")
        kept = [r for r in last_results if r.kept]
        discarded = [r for r in last_results if not r.kept]
        for res in kept:
            user_parts.append(f"BEST Experiment {res.branch_id}:\n")
            user_parts.append(f"  Metrics: {_format_metrics(res.aggregated)}\n")
            if res.hypothesis:
                user_parts.append(f"  Hypothesis: {res.hypothesis}\n")
            if res.experiment:
                user_parts.append(f"  Experiment: {res.experiment}\n")
        if discarded:
            user_parts.append(f"({len(discarded)} other experiments discarded)\n")

    if vision_feedback:
        user_parts.append("\n\n## Vision Comparison\n")
        # Cap vision feedback to avoid huge context
        user_parts.append(vision_feedback[:3000] + ("..." if len(vision_feedback) > 3000 else ""))

    if roundtrip_feedback:
        user_parts.append("\n\n## Per-Image Results (sorted worst → best by DINO)\n")
        # Cap roundtrip feedback — full captions already included for worst images
        user_parts.append(roundtrip_feedback[:4000] + ("..." if len(roundtrip_feedback) > 4000 else ""))

    if caption_diffs:
        user_parts.append(f"\n\n{caption_diffs}")

    # Dedup: show previously proposed hypotheses for this iteration
    if already_proposed:
        user_parts.append("\n\n## Already Proposed This Iteration (propose something DIFFERENT)\n")
        for i, hyp in enumerate(already_proposed, 1):
            user_parts.append(f"{i}. {hyp}\n")

    has_feedback = vision_feedback or roundtrip_feedback
    instruction = (
        "\n\nPropose an improved template. Review the Knowledge Base, then formulate a hypothesis "
        "that builds on previous insights (reference H-ids). Update open problems."
    )
    if has_feedback:
        instruction += (
            " Use the vision comparison and per-image results to ground your hypothesis in specific evidence."
        )
    user_parts.append(instruction)

    user = "".join(user_parts)

    # Log context size for debugging latency
    word_count = len(user.split())
    logger.info("Requesting experiment proposal (%s) — context: ~%d words", model, word_count)

    text = await client.call(model=model, system=system, user=user, max_tokens=16000)

    new_template = _parse_template(text)
    analysis_text = _parse_analysis(text)
    template_changes_description = _parse_template_changes(text)
    should_stop = _parse_converged(text)
    hypothesis = _parse_hypothesis(text)
    experiment = _parse_experiment(text)
    lessons = _parse_lessons(text)
    builds_on = _parse_builds_on(text)
    open_problems = _parse_open_problems(text)

    if not new_template.sections:
        logger.warning("Refined template has no sections — falling back to current template")
        new_template = current_template

    return (
        new_template,
        analysis_text,
        template_changes_description,
        should_stop,
        hypothesis,
        experiment,
        lessons,
        builds_on,
        open_problems,
    )


async def synthesize_templates(
    experiments: list[IterationResult],
    style_profile: StyleProfile,
    *,
    client: ReasoningClient,
    model: str,
) -> tuple[PromptTemplate, str]:
    """Merge the best aspects of multiple experiments into one template.

    Takes the top-performing experiments and asks Claude to combine their
    strongest sections into a single template. Returns (merged_template, hypothesis).
    """

    system = (
        "You are an expert art director. You are given several meta-prompt templates that were "
        "tested in parallel. Each produced different strengths — some improved color accuracy, "
        "others improved composition or character fidelity.\n\n"
        "Your task: MERGE the best aspects of each template into one combined template.\n\n"
        "Rules:\n"
        "- For each section, pick the best version from the experiments based on per-image scores.\n"
        "- If two experiments improved different aspects, combine their strengths.\n"
        "- Do NOT average or water down — pick the strongest phrasing for each section.\n"
        "- Keep the template 6-10 sections, 200-400 words rendered.\n\n"
        "Response format:\n"
        "<rationale>Which sections you took from which experiment and why</rationale>\n"
        "<template>\n"
        '  <section name="..." description="...">value</section>\n'
        "  ...\n"
        "  <negative>things to avoid</negative>\n"
        "</template>"
    )

    user_parts: list[str] = [
        "## Style Profile\n",
        _format_style_profile(style_profile, compact=True),
        "\n\n## Experiments to Merge\n\n",
    ]

    for exp in experiments:
        kept_label = "BEST" if exp.kept else "IMPROVED"
        user_parts.append(f"### Experiment {exp.branch_id} [{kept_label}]\n")
        user_parts.append(f"Hypothesis: {exp.hypothesis}\n")
        user_parts.append(f"Metrics:\n{_format_metrics(exp.aggregated)}\n")
        user_parts.append(f"Template:\n{_format_template(exp.template)}\n\n")

    user_parts.append(
        "Merge these into one template that combines the strongest aspects of each. "
        "Focus on sections where specific experiments showed clear metric advantages."
    )

    user = "".join(user_parts)

    logger.info("Requesting template synthesis (%s)", model)

    text = await client.call(model=model, system=system, user=user, max_tokens=16000)

    merged = _parse_template(text)
    rationale = ""
    rationale_match = re.search(r"<rationale>(.*?)</rationale>", text, re.DOTALL)
    if rationale_match:
        rationale = rationale_match.group(1).strip()

    if not merged.sections:
        logger.warning("Synthesis produced no sections — falling back to best experiment's template")
        merged = experiments[0].template

    hypothesis = (
        f"Synthesis: combining best sections from experiments {', '.join(str(e.branch_id) for e in experiments)}"
    )
    if rationale:
        hypothesis += f" — {rationale[:200]}"

    return merged, hypothesis

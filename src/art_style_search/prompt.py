"""Claude API calls for meta-prompt optimization.

Claude operates on both the prompt template (structure/sections) and the prompt
values (content within sections).  It proposes initial diverse templates and
iteratively refines them based on evaluation metrics.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from art_style_search.types import (
    AggregatedMetrics,
    IterationResult,
    KnowledgeBase,
    PromptSection,
    PromptTemplate,
    ReviewResult,
    StyleProfile,
    classify_hypothesis,
    composite_score,
    get_category_names,
)
from art_style_search.utils import ReasoningClient

if TYPE_CHECKING:
    from art_style_search.experiment import ExperimentProposal

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


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

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
_NEGATIVE_RE = re.compile(r"<negative>(.*?)</negative>", re.DOTALL)
_CAPTION_SECTIONS_RE = re.compile(r"<caption_sections>(.*?)</caption_sections>", re.DOTALL)
_CAPTION_LENGTH_RE = re.compile(r"<caption_length>\s*(\d+)\s*</caption_length>", re.DOTALL)
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
_CHANGED_SECTION_RE = re.compile(r"<changed_section>(.*?)</changed_section>", re.DOTALL)
_TARGET_CATEGORY_RE = re.compile(r"<target_category>(.*?)</target_category>", re.DOTALL)


def _parse_template(text: str) -> PromptTemplate:
    """Extract a PromptTemplate from Claude's XML-style response.

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

    negative = None
    neg_match = _NEGATIVE_RE.search(text)
    if neg_match:
        negative = neg_match.group(1).strip() or None

    caption_sections: list[str] = []
    cs_match = _CAPTION_SECTIONS_RE.search(text)
    if cs_match:
        caption_sections = [s.strip() for s in cs_match.group(1).split(",") if s.strip()]

    caption_length_target = 0
    cl_match = _CAPTION_LENGTH_RE.search(text)
    if cl_match:
        caption_length_target = int(cl_match.group(1))

    return PromptTemplate(
        sections=sections,
        negative_prompt=negative,
        caption_sections=caption_sections,
        caption_length_target=caption_length_target,
    )


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


@dataclass
class RefinementResult:
    """Complete result of a template refinement by Claude."""

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
    confirmed = _CONFIRMED_RE.search(text)
    rejected = _REJECTED_RE.search(text)
    insight = _NEW_INSIGHT_RE.search(text)
    return Lessons(
        confirmed=confirmed.group(1).strip() if confirmed else "",
        rejected=rejected.group(1).strip() if rejected else "",
        new_insight=insight.group(1).strip() if insight else "",
    )


def _parse_changed_section(text: str) -> str:
    """Extract the <changed_section> tag — returns section name or empty string."""
    m = _CHANGED_SECTION_RE.search(text)
    return m.group(1).strip() if m else ""


def _parse_target_category(text: str) -> str:
    """Extract the <target_category> tag — returns category name or empty string."""
    m = _TARGET_CATEGORY_RE.search(text)
    return m.group(1).strip() if m else ""


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
        "HOW to describe/caption reference images.\n\n"
        "## Goal\n"
        "The captions produced must serve a DUAL purpose:\n"
        "1. Contain enough detail to RECREATE the original image (we measure this with metrics).\n"
        "2. Embed REUSABLE art-style guidance — labeled sections with style rules, color palette, "
        "technique descriptions, etc. — that can later be applied to generate NEW art in the same style "
        "with different subjects.\n\n"
        "## How the system works\n"
        "meta-prompt + reference image → Gemini Pro caption → Gemini Flash generation → compare with original.\n"
        "The meta-prompt is the ONLY thing being optimized. It must instruct the captioner "
        "to describe every detail needed for faithful recreation AND embed the art-style guidance "
        "from the Style Profile into every caption as reusable style rules.\n\n"
        "## Meta-prompt requirements\n"
        "- 8-15 sections, each instructing the captioner WHAT to describe and HOW precisely.\n"
        "- Must cover: technique/medium, colors, composition, characters/figures, "
        "background/environment, textures/details, lighting, mood/atmosphere.\n"
        "- Sections should EMBED the core style rules from the Style Profile as literal text — "
        "the captioner should weave these rules into every caption as a shared style foundation, "
        "then add per-image observations on top.\n"
        "- Each section should be 4-8 sentences of instruction.\n"
        "- Total rendered meta-prompt should be 1200-1800 words.\n"
        "- Include a negative section: what the captioner should tell the generator to AVOID.\n"
        "- The meta-prompt must produce captions specific enough that someone who has never "
        "seen the image could recreate it from the caption alone.\n\n"
        "## MANDATORY: Style Foundation section\n"
        "Every template MUST include a section named 'style_foundation' as the FIRST section. "
        "This section instructs the captioner to open every caption with an [Art Style] block "
        "containing FIXED, REUSABLE style rules copied verbatim from the Style Profile. "
        "The [Art Style] block must be nearly IDENTICAL across all captions — it is the shared "
        "style DNA that enables generating new art in the same style with different subjects. "
        "DO NOT remove, rename, merge, or weaken this section even if recreation metrics dip — "
        "it is a hard constraint, not subject to optimization.\n\n"
        "## Caption output structure\n"
        "The meta-prompt must instruct the captioner to produce captions with LABELED SECTIONS. "
        "The FIRST section must always be [Art Style] (the shared style block). "
        "You decide the remaining sections and their order — that is part of experimentation.\n"
        "- Specify the caption output sections as a <caption_sections> tag (comma-separated list). "
        "The first entry MUST be 'Art Style'.\n"
        "- Specify the target caption length as a <caption_length> tag (word count).\n"
        "- The [Art Style] section should be IDENTICAL across captions (shared style rules). "
        "All other sections contain per-image specific observations.\n"
        "- A style_consistency metric measures how similar the [Art Style] blocks are across "
        "captions — higher consistency is rewarded in the composite score.\n\n"
        "## Example of a good meta-prompt section\n"
        '<section name="colors_and_palette" description="instruct captioner on color description '
        'with embedded style rules">'
        "This art style uses a warm earth-tone palette dominated by burnt sienna, raw umber, "
        "and cadmium yellow, with cool accents of cerulean blue. Saturation is moderate — colors "
        "feel muted and aged rather than vibrant. "
        "When describing the image, note the EXACT colors visible using specific color names "
        "(not just 'brown' or 'yellow'). Describe the overall color temperature, saturation levels, "
        "and how colors relate to each other. Note any gradients, color transitions, or areas where "
        "the palette deviates from the core warm-earth foundation. "
        "In your [Color Palette] section, first state the core style palette rules, then describe "
        "how this specific image applies or varies from them."
        "</section>\n\n"
        "## Diversity across meta-prompts\n"
        "- Vary the set of caption output section names and their ordering.\n"
        "- Vary caption length targets (e.g. 400, 600, 800 words).\n"
        "- Vary emphasis: some focus on technique precision, others on spatial accuracy, "
        "others on mood fidelity.\n"
        "- Vary the balance between shared style guidance vs per-image detail.\n"
        "- Vary instruction style: some give the captioner strict checklists, "
        "others give artistic direction, others ask for technical analysis.\n"
        "- All must be comprehensive — diversity is in approach, not coverage.\n\n"
        f"Produce exactly {num_branches} meta-prompts, each wrapped in a <branch> tag.\n\n"
        "Response format (repeat for each branch):\n"
        "<branch>\n"
        "<template>\n"
        '  <section name="..." description="what this instructs the captioner to describe">'
        "instruction for the captioner with embedded style rules (4-8 sentences)</section>\n"
        "  ... (8-15 sections)\n"
        "  <negative>instruct captioner to tell generator what to avoid</negative>\n"
        "  <caption_sections>Art Style, Color Palette, Composition, ...</caption_sections>\n"
        "  <caption_length>500</caption_length>\n"
        "</template>\n"
        "</branch>"
    )

    user = (
        "Based on the following style profile of the reference images, create the initial "
        "meta-prompts. Remember: these are INSTRUCTIONS for a captioner, not direct image prompts.\n\n"
        f"{_format_style_profile(style_profile)}"
    )

    logger.info("Requesting %d initial templates (%s)", num_branches, model)

    text = await client.call(model=model, system=system, user=user, max_tokens=24000)

    templates = _parse_initial_templates(text, num_branches)

    for i, t in enumerate(templates):
        if not t.sections:
            logger.warning("Branch %d initial template has no sections — raw response may need review", i)

    return templates


def _parse_refinement_branches(text: str, num_experiments: int) -> list[RefinementResult]:
    """Parse multiple RefinementResults from <branch>-wrapped response blocks.

    Each branch is expected to contain the same response tags as a single
    ``refine_template`` call (hypothesis, experiment, template, etc.).
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


def enforce_hypothesis_diversity(
    results: list[RefinementResult],
    template: PromptTemplate,
) -> list[RefinementResult]:
    """Deduplicate experiments targeting the same category. Keep the first occurrence."""
    seen_categories: set[str] = set()
    diverse_results: list[RefinementResult] = []
    category_names = get_category_names(template)

    for r in results:
        # Prefer Claude's explicit target_category; fall back to keyword classification
        cat = r.target_category if r.target_category else classify_hypothesis(r.hypothesis, category_names)
        if cat in seen_categories:
            logger.warning(
                "Dropping duplicate-category experiment (category=%s): %s",
                cat,
                r.hypothesis[:80],
            )
            continue
        seen_categories.add(cat)
        diverse_results.append(r)

    if len(diverse_results) < len(results):
        logger.info("Diversity filter: kept %d/%d experiments", len(diverse_results), len(results))
    return diverse_results


async def propose_experiments(
    style_profile: StyleProfile,
    current_template: PromptTemplate,
    knowledge_base: KnowledgeBase,
    best_metrics: AggregatedMetrics | None,
    last_results: list[IterationResult] | None,
    *,
    client: ReasoningClient,
    model: str,
    num_experiments: int,
    vision_feedback: str = "",
    roundtrip_feedback: str = "",
    caption_diffs: str = "",
) -> list[RefinementResult]:
    """Propose N experiments in a single Claude call.

    Uses ``<branch>`` tags so Claude generates all experiments at once,
    ensuring inherent diversity without sequential dedup.  Follows the
    same pattern as :func:`propose_initial_templates`.
    """

    system = (
        "You are an expert art director and prompt engineer optimizing a META-PROMPT.\n\n"
        "## How this system works\n"
        "You are NOT writing image generation prompts directly. You are writing a META-PROMPT — "
        "instructions that tell Gemini Pro HOW to caption/describe reference images. "
        "Those captions are then used by Gemini Flash to generate images. "
        "The goal is to produce captions precise enough to RECREATE the original images.\n\n"
        "The pipeline: meta-prompt + reference image → Gemini Pro caption → Gemini Flash generation → compare with original.\n\n"
        "## Dual-purpose captions\n"
        "The captions produced must serve TWO purposes:\n"
        "1. Contain enough detail to faithfully recreate the reference image (measured by metrics).\n"
        "2. Embed REUSABLE art-style guidance — labeled sections with style rules, color palette, "
        "technique descriptions, mood cues, etc. — that can later generate NEW art in the same style.\n"
        "The meta-prompt must embed core style rules from the Style Profile as literal text "
        "that the captioner weaves into every caption as a shared style foundation, "
        "plus per-image specific observations.\n\n"
        "## What makes a good meta-prompt\n"
        "The meta-prompt must instruct the captioner to describe EVERY visual detail needed "
        "for faithful recreation, while embedding style guidance:\n"
        "- Exact colors, technique, medium, brushwork — with style rules embedded\n"
        "- Character/figure details: poses, expressions, clothing, proportions, identity\n"
        "- Background/environment: setting, architecture, nature elements\n"
        "- Composition: layout, framing, depth, perspective — with style patterns noted\n"
        "- Lighting, shadows, atmospheric effects — with style lighting rules\n"
        "- Textures, patterns, fine details — with technique guidance\n"
        "- Mood, emotional tone — with style mood rules\n"
        "- What to AVOID (common AI generation artifacts)\n\n"
        "The meta-prompt should be 8-15 sections, each 4-8 sentences of instruction "
        "with embedded style rules from the Style Profile. "
        "Total rendered prompt should be 1200-1800 words.\n\n"
        "## MANDATORY: Style Foundation section\n"
        "The template MUST include a section named 'style_foundation' as the FIRST section. "
        "This section instructs the captioner to open every caption with an [Art Style] block "
        "containing FIXED, REUSABLE style rules from the Style Profile. "
        "The [Art Style] block must be nearly IDENTICAL across all captions. "
        "DO NOT remove, rename, merge, or weaken this section even if recreation metrics dip — "
        "it is a hard constraint, not subject to optimization. "
        "A style_consistency metric measures cross-caption similarity of [Art Style] blocks.\n\n"
        "## Caption output structure\n"
        "The meta-prompt must instruct the captioner to produce captions with LABELED SECTIONS. "
        "The FIRST section must always be [Art Style] (the shared style block). "
        "You decide the remaining sections and their order — that is part of experimentation.\n"
        "- Specify caption sections as <caption_sections> and target length as <caption_length>.\n"
        "- The first entry in <caption_sections> MUST be 'Art Style'.\n"
        "- The [Art Style] section should be IDENTICAL across captions (shared style rules). "
        "All other sections contain per-image specific observations.\n\n"
        "## Metric guidance\n"
        "Per-image metrics (each generated image vs its paired original):\n"
        "- DreamSim similarity (higher=better): human-aligned perceptual similarity that captures semantic "
        "content, structural layout, color, and mid-level features (pose, composition). "
        "0.4=somewhat similar, 0.6=good reproduction, 0.8+=very close match.\n"
        "- Color histogram (higher=better): color palette match. 0.7=similar, 0.9+=very close.\n"
        "- Texture (higher=better): Gabor filter energy similarity for brush strokes/patterns. 0.7=similar, 0.9+=very close.\n"
        "- SSIM (higher=better): pixel-level structural similarity. 0.5=moderate, 0.7=good, 0.9+=near-identical.\n"
        "- HPS v2 (higher=better): caption-image alignment. Range 0.20-0.30.\n"
        "- Aesthetics (higher=better, 1-10): visual quality. 5=mediocre, 7=good, 8+=excellent.\n"
        "Per-image vision scores (from Gemini visual comparison, ternary: MATCH=1.0, PARTIAL=0.5, MISS=0.0):\n"
        "- vision_style: art technique reproduction (aggregated as ratio of images matching).\n"
        "- vision_subject: character/subject fidelity.\n"
        "- vision_composition: spatial layout accuracy.\n"
        "Weights are ADAPTIVE — metrics with more variance across experiments get higher weight.\n\n"
        "## Iteration strategy\n"
        f"- Propose exactly {num_experiments} experiments, each in a <branch> tag. "
        "Each MUST have a DIFFERENT hypothesis targeting a DIFFERENT weakness or category.\n"
        "- There are no fixed 'branches' — shift focus freely between categories "
        "as the weakest area changes.\n"
        "- Make EXACTLY 1 section change per experiment — modify a single section's value. "
        "This enables clean attribution of which change helped or hurt.\n"
        "- Experiments can vary: section content, caption output section names/ordering, "
        "caption length target, balance of shared style vs per-image detail.\n"
        "- If DreamSim is weak: the captions miss structural, color, or semantic details — "
        "add instructions for the captioner to be more specific about those.\n"
        "- If per-image scores vary widely: some images are harder — consider "
        "conditional captioning instructions (e.g. 'for character images describe X; "
        "for backgrounds describe Y').\n"
        "- Use the vision comparison and per-image roundtrip feedback to identify "
        "what the captions consistently miss.\n"
        "- CRITICAL: Read the Knowledge Base carefully. Under Per-Category Status, "
        "'Last rejected' entries show failed approaches — do NOT repeat them. "
        "Build on confirmed insights. Reference hypothesis IDs (e.g. 'builds on H3').\n"
        "- Use Per-Category Status to identify which style dimensions need work.\n"
        "- Target the weakest category or build on partially confirmed hypotheses.\n"
        "- Use the Open Problems list to focus on the highest-priority gaps.\n"
        "- Update <open_problems> each iteration: add new ones, remove solved ones, re-rank.\n\n"
        "## Optimization dynamics\n"
        "Apply these principles when proposing changes:\n\n"
        "**Momentum**: The Knowledge Base contains confirmed insights from prior iterations. "
        "These are VALIDATED improvements — double down on them. If a confirmed insight "
        "improved one aspect, explore whether the same principle applies to other sections. "
        "Do not revisit or undo confirmed improvements unless metrics specifically regressed.\n\n"
        "**Step size**: Adapt the magnitude of your changes to the current score level:\n"
        "- When composite score is LOW (<0.35): make BOLD changes — restructure sections, "
        "try very different instruction styles, experiment with caption length.\n"
        "- When composite score is MODERATE (0.35-0.50): make TARGETED changes — refine "
        "specific wording, adjust emphasis within sections, fine-tune constraints.\n"
        "- When composite score is HIGH (>0.50): make SURGICAL changes — tweak individual "
        "phrases, adjust quantitative thresholds, polish specific failure modes. "
        "Small changes matter more here; large changes risk regression.\n\n"
        "**Diversity pressure**: Each experiment in this batch MUST target a different "
        "hypothesis category. If a category has 3+ rejected approaches with no confirmed "
        "insights, DEPRIORITIZE it — focus effort where confirmed partial improvements "
        "suggest further gains are possible.\n\n"
        "If metrics have plateaued, append [CONVERGED] at the very end of the LAST branch.\n\n"
        f"Response format — exactly {num_experiments} branches, each containing ALL required tags:\n"
        "<branch>\n"
        "<lessons>\n"
        "  <confirmed>Which previous hypotheses are confirmed by THIS iteration's results?</confirmed>\n"
        "  <rejected>Which previous hypotheses are rejected? What didn't work and why?</rejected>\n"
        "  <new_insight>Any new observation from the data not covered by existing hypotheses</new_insight>\n"
        "</lessons>\n"
        "<hypothesis>Based on the knowledge base and current results, what is the "
        "PRIMARY remaining gap? Be specific — name the metric, the images, the visual element.</hypothesis>\n"
        "<builds_on>H-ids this builds on, or 'none' for fresh direction</builds_on>\n"
        "<experiment>The specific change you're making to test this hypothesis</experiment>\n"
        "<changed_section>name of the SINGLE section you modified</changed_section>\n"
        "<target_category>the primary category this experiment targets (must be unique across branches)</target_category>\n"
        "<open_problems>\n"
        "  1. Most critical remaining problem\n"
        "  2. Second most critical\n"
        "  3. Third (if any)\n"
        "</open_problems>\n"
        "<template_changes>structural changes or 'none'</template_changes>\n"
        "<template>\n"
        '  <section name="..." description="...">value with embedded style rules (4-8 sentences)</section>\n'
        "  ... (8-15 sections)\n"
        "  <negative>things to avoid</negative>\n"
        "  <caption_sections>ordered comma-separated list of labeled output sections</caption_sections>\n"
        "  <caption_length>target word count for captions</caption_length>\n"
        "</template>\n"
        "</branch>\n"
        "(repeat for each experiment)\n"
        "[CONVERGED]  (only if converged, after the last branch)"
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
        score = composite_score(best_metrics)
        regime = "LOW" if score < 0.35 else "MODERATE" if score < 0.50 else "HIGH"
        user_parts.append(f"\nCurrent composite score: {score:.4f} ({regime} regime)\n")

    # Knowledge base — structured lessons from all previous experiments
    kb_text = knowledge_base.render_for_claude()
    if kb_text:
        user_parts.append("\n\n")
        user_parts.append(kb_text)

    # Suggest target categories for diversity
    if knowledge_base and knowledge_base.hypotheses:
        category_names = get_category_names(current_template)
        suggested = knowledge_base.suggest_target_categories(num_experiments, category_names)
        if suggested:
            user_parts.append(
                "\n## Suggested Target Categories (ranked by improvement potential)\n"
                + "\n".join(f"{i}. {cat}" for i, cat in enumerate(suggested, 1))
            )

    # Show last iteration results — only the kept experiment in detail
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
            # Show worst experiment for negative learning
            worst = min(discarded, key=lambda r: composite_score(r.aggregated))
            worst_parts: list[str] = ["\n## Worst Experiment (learn from this failure)\n"]
            if worst.hypothesis:
                worst_parts.append(f"Hypothesis: {worst.hypothesis}\n")
            if worst.experiment:
                worst_parts.append(f"Experiment: {worst.experiment}\n")
            worst_parts.append(f"Metrics:\n{_format_metrics(worst.aggregated)}\n")
            if worst.per_image_scores and worst.iteration_captions:
                idx = min(
                    range(len(worst.per_image_scores)),
                    key=lambda i: worst.per_image_scores[i].dreamsim_similarity,
                )
                if idx < len(worst.iteration_captions):
                    cap = worst.iteration_captions[idx]
                    cap_words = cap.text.split()
                    cap_text = " ".join(cap_words[:150]) + ("..." if len(cap_words) > 150 else "")
                    worst_parts.append(
                        f"Worst image ({cap.image_path.name}): "
                        f"DS={worst.per_image_scores[idx].dreamsim_similarity:.3f}\n"
                        f"Caption: {cap_text}\n"
                    )
            if worst.vision_feedback:
                vf_words = worst.vision_feedback.split()
                vf = " ".join(vf_words[:100]) + ("..." if len(vf_words) > 100 else "")
                worst_parts.append(f"Vision feedback: {vf}\n")
            user_parts.append("".join(worst_parts))

    if vision_feedback:
        user_parts.append("\n\n## Vision Comparison (Gemini analysis of generated vs reference images)\n")
        # Cap vision feedback to ~500 words to prevent context degradation
        vision_words = vision_feedback.split()
        if len(vision_words) > 500:
            user_parts.append(" ".join(vision_words[:500]) + "\n[...truncated]")
        else:
            user_parts.append(vision_feedback)

    if roundtrip_feedback:
        user_parts.append("\n\n## Per-Image Results (sorted worst → best by DreamSim)\n")
        # Cap roundtrip feedback to ~800 words (full detail for worst images, metrics-only for rest)
        roundtrip_words = roundtrip_feedback.split()
        if len(roundtrip_words) > 800:
            user_parts.append(" ".join(roundtrip_words[:800]) + "\n[...truncated]")
        else:
            user_parts.append(roundtrip_feedback)

    if caption_diffs:
        user_parts.append(f"\n\n{caption_diffs}")

    has_feedback = vision_feedback or roundtrip_feedback
    instruction = (
        f"\n\nPropose {num_experiments} improved templates, each in a <branch> tag. "
        "Each experiment must target a DIFFERENT weakness — review the Knowledge Base, "
        "then formulate hypotheses that build on previous insights (reference H-ids). "
        "Update open problems in each branch."
    )
    if has_feedback:
        instruction += (
            " Use the vision comparison and per-image results to ground your hypotheses in specific evidence."
        )
    user_parts.append(instruction)

    user = "".join(user_parts)

    logger.info(
        "Requesting %d experiment proposals (%s) — context: ~%d words", num_experiments, model, len(user.split())
    )

    text = await client.call(model=model, system=system, user=user, max_tokens=30000)

    results = _parse_refinement_branches(text, num_experiments)

    # Check for convergence signal after all branches (top-level [CONVERGED])
    if _parse_converged(text):
        if results:
            results[-1].should_stop = True
        else:
            # No valid branches but converged — return a dummy result
            results.append(
                RefinementResult(
                    template=current_template,
                    analysis="",
                    template_changes="",
                    should_stop=True,
                    hypothesis="",
                    experiment="",
                    lessons=Lessons(),
                    builds_on=None,
                    open_problems=[],
                )
            )

    if not results:
        logger.warning("No valid experiments parsed from response")

    return results


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
        "- Keep the template 8-15 sections, 1200-1800 words rendered.\n"
        "- Preserve embedded style rules in section values.\n"
        "- MANDATORY: The first section must be 'style_foundation' with fixed style rules. "
        "The first entry in caption_sections must be 'Art Style'. Never remove these.\n"
        "- Merge caption output structure: pick the best caption_sections ordering and caption_length "
        "from the experiments, or combine them.\n\n"
        "Response format:\n"
        "<rationale>Which sections you took from which experiment and why</rationale>\n"
        "<template>\n"
        '  <section name="..." description="...">value</section>\n'
        "  ...\n"
        "  <negative>things to avoid</negative>\n"
        "  <caption_sections>ordered comma-separated list of labeled output sections</caption_sections>\n"
        "  <caption_length>target word count for captions</caption_length>\n"
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

    text = await client.call(model=model, system=system, user=user, max_tokens=12000)

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


# ---------------------------------------------------------------------------
# Independent review (CycleResearcher-inspired)
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM = (
    "You are a critical scientific reviewer evaluating prompt optimization experiments.\n"
    "Your role is INDEPENDENT from the proposer — be skeptical and evidence-based.\n\n"
    "For each experiment, assess:\n"
    "1. Did the metric changes actually support the hypothesis? (Many metric movements are noise)\n"
    "2. Was the hypothesis specific enough to be falsifiable?\n"
    "3. Did the changed section actually cause the observed effects, or could it be coincidence?\n"
    "4. Are there confounding factors (e.g., caption length changed alongside content)?\n\n"
    "Then provide strategic guidance:\n"
    "- Which categories have the most room for improvement based on metric gaps?\n"
    "- Are there hypotheses that keep getting proposed but never confirmed? Flag them.\n"
    "- What is the single most impactful thing the next iteration should try?\n\n"
    "Be brutally honest. A confirmed improvement of +0.005 on one metric while others stayed flat "
    "is likely noise, not a real improvement. Look for consistent patterns across multiple metrics.\n\n"
    "Respond with:\n"
    "<assessments>\nOne paragraph per experiment: [EXP_ID] SIGNAL|NOISE|MIXED — explanation\n</assessments>\n"
    "<noise_vs_signal>Overall analysis of which metric movements are real</noise_vs_signal>\n"
    "<strategic_guidance>What next iteration should focus on</strategic_guidance>\n"
    "<recommended_categories>comma-separated list of categories to target</recommended_categories>"
)


async def review_iteration(
    experiments: list[IterationResult],
    proposals: list[ExperimentProposal],
    baseline_metrics: AggregatedMetrics | None,
    knowledge_base: KnowledgeBase,
    *,
    client: ReasoningClient,
    model: str,
) -> ReviewResult:
    """Independent review of iteration results — skeptical assessment of improvements."""

    user_parts: list[str] = []
    user_parts.append("## Experiments to Review\n")

    for exp, prop in zip(experiments, proposals, strict=False):
        m = exp.aggregated
        user_parts.append(
            f"### Experiment {exp.branch_id}\n"
            f"Hypothesis: {prop.hypothesis}\n"
            f"Changed section: {prop.experiment_desc}\n"
            f"Kept: {exp.kept}\n"
            f"Metrics: {_format_metrics(m)}\n"
        )
        if baseline_metrics:
            delta_ds = m.dreamsim_similarity_mean - baseline_metrics.dreamsim_similarity_mean
            delta_color = m.color_histogram_mean - baseline_metrics.color_histogram_mean
            delta_hps = m.hps_score_mean - baseline_metrics.hps_score_mean
            user_parts.append(f"Deltas vs baseline: DS={delta_ds:+.4f} Color={delta_color:+.4f} HPS={delta_hps:+.4f}\n")

    if baseline_metrics:
        user_parts.append(f"\n## Baseline Metrics\n{_format_metrics(baseline_metrics)}\n")

    kb_text = knowledge_base.render_for_claude(max_words=500)
    if kb_text:
        user_parts.append(f"\n{kb_text}\n")

    user = "\n".join(user_parts)
    text = await client.call(model=model, system=_REVIEW_SYSTEM, user=user, max_tokens=6000)

    # Parse response
    assessments_match = re.search(r"<assessments>(.*?)</assessments>", text, re.DOTALL)
    noise_match = re.search(r"<noise_vs_signal>(.*?)</noise_vs_signal>", text, re.DOTALL)
    guidance_match = re.search(r"<strategic_guidance>(.*?)</strategic_guidance>", text, re.DOTALL)
    cats_match = re.search(r"<recommended_categories>(.*?)</recommended_categories>", text, re.DOTALL)

    assessments_raw = assessments_match.group(1).strip() if assessments_match else ""
    experiment_assessments = [line.strip() for line in assessments_raw.split("\n") if line.strip()]

    return ReviewResult(
        experiment_assessments=experiment_assessments,
        noise_vs_signal=noise_match.group(1).strip() if noise_match else "",
        strategic_guidance=guidance_match.group(1).strip() if guidance_match else "",
        recommended_categories=[
            c.strip() for c in (cats_match.group(1).strip().split(",") if cats_match else []) if c.strip()
        ],
    )

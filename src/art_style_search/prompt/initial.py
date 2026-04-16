"""Zero-step: brainstorm initial-template sketches, rank them, then expand survivors in parallel."""

from __future__ import annotations

import asyncio
import logging

from art_style_search.contracts import InitialTemplateSketch
from art_style_search.prompt._format import _format_style_profile
from art_style_search.prompt.json_contracts import (
    response_schema,
    schema_hint,
    validate_initial_brainstorm_payload,
    validate_initial_expansion_payload,
    validate_ranking_payload,
)
from art_style_search.types import PromptTemplate, StyleProfile
from art_style_search.utils import ReasoningClient

logger = logging.getLogger(__name__)


_BRAINSTORM_EXAMPLE = (
    "## Example of a good sketch\n"
    '{"approach_summary":"subject-first strict checklist",'
    '"emphasis":"technique","instruction_style":"checklist","caption_length_target":3000,'
    '"caption_sections":["Art Style","Subject","Color Palette","Composition","Lighting & Atmosphere"],'
    '"distinguishing_feature":"Terse imperative bullets per section; subject facets enumerated '
    '(species, clothing, pose, expression, props) to force identity specificity over style language."}\n\n'
    "## Example of a bad sketch (too vague — avoid this)\n"
    '{"approach_summary":"comprehensive","emphasis":"general","instruction_style":"detailed",'
    '"caption_length_target":4000,"caption_sections":["Art Style","Subject"],'
    '"distinguishing_feature":"better than baseline"}'
)


_BASE_REQUIREMENTS = (
    "## Goal\n"
    "We are designing META-PROMPTS — instructions that tell Gemini Pro HOW to caption reference images. "
    "Captions serve TWO purposes: (1) recreate the image faithfully, (2) embed REUSABLE art-style guidance "
    "in labeled sections so the same style can be applied to new subjects.\n\n"
    "Pipeline: meta-prompt + reference -> Gemini Pro caption -> Gemini Flash generation -> compare with original.\n\n"
    "## NON-NEGOTIABLE structural rules (every expanded template must obey)\n"
    "1. First section MUST be 'style_foundation' — produces the [Art Style] block (shared style DNA, "
    "repeated inside every caption and measured by style_consistency).\n"
    "2. Second section MUST be 'subject_anchor' — produces the [Subject] block (identity, features, pose; "
    "most important for reproduction).\n"
    "3. caption_sections MUST start with ['Art Style', 'Subject', ...].\n"
    "4. Total rendered template MUST be 2000-8000 words across 8-20 sections.\n"
    "5. 'style_foundation.value' MUST realize the 5-slot [Art Style] SKELETON — RULES ONLY that apply to every "
    "image in this style, never per-image observations. Start with the literal 'How to Draw:' marker, then: "
    "Shading & Light (layer stack + edge softness + key-fill-rim), Color Principle (generic palette families + "
    "value + saturation, no named colors), Surface & Texture (grain + class-appropriate material vocabulary, "
    "no specific objects), and Style Invariants (3-5 generative MUST/NEVER rules — e.g. 'MUST: every character "
    "has exactly one exaggerated feature'; 'NEVER: more than three saturated hues in one frame'). "
    "Genre labels like '3D CGI of X', 'cel-shaded anime', or '{Artist}-style' are forbidden inside [Art Style]. "
    "Character proportions (heads-tall + archetype) belong in subject_anchor, NOT here.\n"
    "6. 'subject_anchor.value' MUST include a 'Proportions:' sub-block requiring numeric head-heights-tall AND "
    "an archetype phrase (one of chibi / stylized-youth / heroic / realistic-adult / elongated).\n"
    "7. The captioner must use MEDIUM-CLASS discipline: classify every image as exactly one of "
    "A hand-drawn 2D / B vector-flat 2D / C stylized 3D CGI / D photoreal 3D / E mixed-2.5D, and use "
    "class-appropriate vocabulary only. Name the observed class inside style_foundation.\n"
    "8. Be laconic: state each style rule ONCE in style_foundation. Other sections reference rules by name, "
    "not by restatement — duplication adds words without adding signal.\n"
)


def _render_sketch(sketch: InitialTemplateSketch, idx: int) -> str:
    return (
        f"### Sketch {idx}\n"
        f"- approach_summary: {sketch.approach_summary}\n"
        f"- emphasis: {sketch.emphasis}\n"
        f"- instruction_style: {sketch.instruction_style}\n"
        f"- caption_length_target: {sketch.caption_length_target}\n"
        f"- caption_sections: {', '.join(sketch.caption_sections)}\n"
        f"- distinguishing_feature: {sketch.distinguishing_feature}\n"
    )


def _brainstorm_system(num_sketches: int) -> str:
    return (
        "You are an expert art director and prompt engineer designing diverse initial meta-prompts.\n\n"
        f"{_BASE_REQUIREMENTS}\n"
        "## Your task — BRAINSTORM ONLY\n"
        f"Produce {num_sketches} short, lightweight sketches that describe DIFFERENT approaches to the meta-prompt. "
        "Do NOT write the full template yet — that happens later. Each sketch is a one-paragraph design intent.\n\n"
        "## Sketch fields (every sketch must have ALL of these)\n"
        "- approach_summary: short phrase naming the variant (e.g. 'subject-first strict checklist')\n"
        "- emphasis: what the variant optimizes for — one of {technique, spatial, mood, balanced, palette}\n"
        "- instruction_style: how the captioner is addressed — one of {checklist, artistic_direction, "
        "technical_analysis, hybrid}\n"
        "- caption_length_target: integer word count target for captions (500-6000, vary across sketches)\n"
        "- caption_sections: ordered list, MUST start with ['Art Style', 'Subject', ...] then 4-10 more "
        "section names of your choosing (these become labeled blocks like [Color Palette] in captions)\n"
        "- distinguishing_feature: 1-2 sentences explaining what makes THIS sketch's approach different from "
        "the others; what mechanism is it betting on?\n\n"
        "## Diversity requirements\n"
        "- Vary caption_length_target meaningfully (some around ~800, some around ~4000)\n"
        "- Vary the caption_sections set/ordering after the two anchors\n"
        "- Vary instruction_style — don't make every sketch a checklist\n"
        "- Vary emphasis — at least 3 different emphasis values across the batch\n"
        "- Each distinguishing_feature must articulate a SPECIFIC mechanism, not a generic 'better' claim\n\n"
        f"{_BRAINSTORM_EXAMPLE}\n\n"
        "## Output format\n"
        f"Return EXACTLY one JSON object with a 'sketches' array of length {num_sketches}. "
        "No markdown fences. No commentary."
    )


def _brainstorm_user(style_profile: StyleProfile, num_sketches: int) -> str:
    return (
        "Based on the following style profile of the reference images, brainstorm "
        f"{num_sketches} diverse meta-prompt sketches.\n\n"
        f"{_format_style_profile(style_profile)}"
    )


def _rank_system() -> str:
    return (
        "You rank initial meta-prompt sketches by expected quality after expansion.\n\n"
        "Rank by these criteria in priority order:\n"
        "1. **Mechanism specificity**: Does the distinguishing_feature articulate a concrete mechanism, "
        "not a vague 'better' claim?\n"
        "2. **Diversity contribution**: Does this sketch contribute approach diversity (instruction_style, "
        "emphasis, caption_length) the others lack?\n"
        "3. **Anchor compatibility**: caption_sections starts with ['Art Style', 'Subject', ...] and the "
        "remaining sections are coherent.\n\n"
        "Return JSON only. No markdown fences. No commentary.\n"
        "Output zero-based indices in best-to-worst order. Include every sketch at most once.\n"
        'Preferred exact wire shape: {"ranked_indices":[2,7,0,5]}'
    )


def _rank_user(sketches: list[InitialTemplateSketch]) -> str:
    parts = ["## Candidate Sketches\n"]
    parts.extend(_render_sketch(sketch, idx) for idx, sketch in enumerate(sketches))
    parts.append(
        f"\n\nValid indices are 0 through {len(sketches) - 1}.\n"
        'Return JSON only in the exact shape {"ranked_indices":[...]}.'
    )
    return "".join(parts)


def _expand_system() -> str:
    return (
        "You are an expert art director. Expand a single initial-template sketch into a complete meta-prompt.\n\n"
        f"{_BASE_REQUIREMENTS}\n"
        "## Your task — EXPAND ONE SKETCH\n"
        "Produce ONE complete PromptTemplate that realizes the sketch's approach. Honor the sketch's "
        "approach_summary, emphasis, instruction_style, caption_length_target, caption_sections, and "
        "distinguishing_feature — these are your design contract.\n\n"
        "## Section requirements\n"
        "- 8-20 sections, with enough detailed instruction to reach 2000-8000 rendered words overall.\n"
        "- First section MUST be 'style_foundation'; it holds the SHARED style DNA the captioner will repeat "
        "inside every caption's [Art Style] block. It MUST realize the 5-slot [Art Style] SKELETON: "
        "(1) How to Draw (medium class A/B/C/D/E + construction + line policy), "
        "(2) Shading & Light (layer stack + edge softness + key-fill-rim), "
        "(3) Color Principle (generic palette families + value + saturation), "
        "(4) Surface & Texture (grain + class-appropriate material vocabulary), "
        "(5) Style Invariants (3-5 generative MUST/NEVER rules that every image in this style obeys). "
        "Start with the literal 'How to Draw:' marker so validation finds it. "
        "OBSERVATIONS-VS-RULES: [Art Style] holds rules only, never per-image observations — no specific body "
        "parts, named objects, proper nouns, actual colors, or pose details. A rule is well-formed only if it "
        "would still be true of a different image in the same style. "
        "Forbid genre labels ('3D CGI of X', 'cel-shaded anime', '{Artist}-style') inside [Art Style].\n"
        "- Second section MUST be 'subject_anchor'; it instructs the captioner to produce a [Subject] block "
        "covering identity/species, distinguishing features, clothing/equipment, pose/action, expression, props. "
        "It MUST include a 'Proportions:' sub-block requiring numeric head-heights-tall and an archetype phrase "
        "(chibi / stylized-youth / heroic / realistic-adult / elongated).\n"
        "- The remaining sections come from the sketch's caption_sections (after Art Style/Subject) — "
        "create one template section per labeled caption section, plus any additional sections needed to "
        "cover technique/medium, lighting, mood, textures, and a negative-instruction section.\n"
        "- [Art Style] targets 400-800 words — concentrated on the 5 skeleton slots of generic RULES, no prose inflation. "
        "[Subject] runs 800-2000 words (it carries the per-image weight, including character proportions). "
        "Ancillary caption sections usually land in 150-400 words.\n"
        "- By default, do NOT include 'Technique' or 'Textures' as caption_sections — their content duplicates "
        "[Art Style]. Propose them as experiments only with a specific mechanism-level reason.\n"
        "- State each style rule ONCE in style_foundation. Ancillary sections reference rules by name "
        "('apply the line-weight rule from style_foundation'), not by verbatim restatement — duplication "
        "inflates word count without adding signal and risks inconsistent rewordings.\n"
        "- STYLE-DNA PURITY (subsumed under the observations-vs-rules rule above, repeated for emphasis): "
        "style_foundation / [Art Style] carries REUSABLE style DNA only. Its palette paragraph MUST name generic "
        "color FAMILIES ('saturated complementary blues', 'warm earth tones with chromatic shadows', 'cadmium "
        "yellow + cerulean + magenta triad'). NEVER let image-specific nouns leak in ('crimson bow tie', "
        "'cadmium yellows of the chest', 'vibrant blue suspenders'). Image-specific palette facts belong in "
        "[Color Palette] or [Subject].\n"
        "- Total rendered template MUST be 2000-8000 words.\n\n"
        "## Output format\n"
        "Return EXACTLY one JSON object — a single PromptTemplate. No markdown fences. No commentary.\n"
        "Required keys: sections (list of {name,description,value}), negative_prompt (string), "
        "caption_sections (list of strings starting with ['Art Style','Subject',...]), "
        "caption_length_target (integer)."
    )


def _expand_user(sketch: InitialTemplateSketch, style_profile: StyleProfile) -> str:
    return (
        "Expand the following sketch into a complete meta-prompt. The sketch defines the approach; "
        "you must realize it in a 2000-8000 word template that obeys all anchor rules.\n\n"
        "## Sketch to Expand\n"
        f"{_render_sketch(sketch, 0)}\n"
        "## Style Profile\n"
        f"{_format_style_profile(style_profile)}"
    )


async def brainstorm_initial_sketches(
    style_profile: StyleProfile,
    *,
    client: ReasoningClient,
    model: str,
    num_sketches: int,
) -> list[InitialTemplateSketch]:
    """Brainstorm short sketches describing diverse meta-prompt approaches (one call)."""
    system = _brainstorm_system(num_sketches)
    user = _brainstorm_user(style_profile, num_sketches)
    logger.info("Brainstorming %d initial-template sketches (%s)", num_sketches, model)
    sketches = await client.call_json(
        model=model,
        system=system,
        user=user,
        validator=lambda data: validate_initial_brainstorm_payload(data, num_sketches=num_sketches),
        response_name="initial_brainstorm",
        schema_hint=schema_hint("initial_brainstorm"),
        response_schema=response_schema("initial_brainstorm"),
        max_tokens=40000,
        repair_retries=2,
        temperature=0.9,
        reasoning_effort="medium",
    )
    if not sketches:
        logger.warning("No valid sketches parsed from initial brainstorm response")
    return sketches


async def rank_initial_sketches(
    sketches: list[InitialTemplateSketch],
    *,
    client: ReasoningClient,
    model: str,
) -> list[InitialTemplateSketch]:
    """Reorder sketches by expected expansion quality. Falls back to original order on failure."""
    if len(sketches) <= 1:
        return list(sketches)
    try:
        ranked_indices = await client.call_json(
            model=model,
            system=_rank_system(),
            user=_rank_user(sketches),
            validator=lambda data: validate_ranking_payload(data, num_sketches=len(sketches)),
            response_name="initial_ranking",
            schema_hint=schema_hint("ranking"),
            response_schema=response_schema("ranking"),
            max_tokens=10000,
            repair_retries=1,
            final_failure_log_level=logging.INFO,
            temperature=0.1,
            reasoning_effort="medium",
        )
    except Exception as exc:
        logger.info("Initial ranking failed; falling back to brainstorm order: %s: %s", type(exc).__name__, exc)
        return list(sketches)
    return [sketches[idx] for idx in ranked_indices]


async def expand_initial_sketches(
    sketches: list[InitialTemplateSketch],
    style_profile: StyleProfile,
    *,
    client: ReasoningClient,
    model: str,
) -> list[PromptTemplate | None]:
    """Expand each sketch into a full template in parallel. Failed expansions return None."""
    if not sketches:
        return []
    system = _expand_system()
    tasks = [
        client.call_json(
            model=model,
            system=system,
            user=_expand_user(sketch, style_profile),
            validator=validate_initial_expansion_payload,
            response_name=f"initial_expansion_{idx}",
            schema_hint=schema_hint("initial_expansion"),
            response_schema=response_schema("initial_expansion"),
            max_tokens=24000,
            repair_retries=2,
            temperature=0.3,
            reasoning_effort="medium",
        )
        for idx, sketch in enumerate(sketches)
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[PromptTemplate | None] = []
    failures = 0
    for raw in raw_results:
        if isinstance(raw, BaseException):
            failures += 1
            logger.warning("Initial expansion failed: %s: %s", type(raw).__name__, raw)
            results.append(None)
            continue
        results.append(raw)
    logger.info("Initial expansion finished: %d/%d succeeded", len(sketches) - failures, len(sketches))
    return results


async def propose_initial_templates(
    style_profile: StyleProfile,
    num_branches: int,
    *,
    client: ReasoningClient,
    model: str,
) -> list[PromptTemplate]:
    """Generate diverse initial prompt templates via brainstorm -> rank -> expand."""
    num_sketches = max(num_branches * 2, num_branches)
    sketches = await brainstorm_initial_sketches(
        style_profile,
        client=client,
        model=model,
        num_sketches=num_sketches,
    )
    if not sketches:
        # Empty placeholders — zero_step._sanitize_initial_templates swaps in the compiled fallback.
        return [
            PromptTemplate(sections=[], negative_prompt=None, caption_sections=[], caption_length_target=0)
        ] * num_branches

    ranked = await rank_initial_sketches(sketches, client=client, model=model)
    top = ranked[:num_branches]
    expansions = await expand_initial_sketches(top, style_profile, client=client, model=model)

    # Pad failed expansions and short batches with empty placeholders; sanitize fills them with the fallback.
    placeholder = PromptTemplate(sections=[], negative_prompt=None, caption_sections=[], caption_length_target=0)
    templates: list[PromptTemplate] = [t if t is not None else placeholder for t in expansions]
    while len(templates) < num_branches:
        templates.append(placeholder)
    return templates[:num_branches]


__all__ = [
    "brainstorm_initial_sketches",
    "expand_initial_sketches",
    "propose_initial_templates",
    "rank_initial_sketches",
]

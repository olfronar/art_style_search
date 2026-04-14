"""Per-iteration experiment proposals: brainstorm sketches, rank them, then expand."""

from __future__ import annotations

import asyncio
import logging

from art_style_search.contracts import ExperimentSketch, Lessons, RefinementResult
from art_style_search.prompt._format import (
    _format_metrics,
    _format_style_profile,
    _format_template,
    _truncate_words,
    format_knowledge_base,
    suggest_target_categories,
)
from art_style_search.prompt.json_contracts import (
    schema_hint,
    validate_brainstorm_payload,
    validate_expansion_payload,
    validate_ranking_payload,
)
from art_style_search.scoring import classify_hypothesis, composite_score
from art_style_search.types import (
    AggregatedMetrics,
    IterationResult,
    KnowledgeBase,
    PromptTemplate,
    StyleProfile,
    get_category_names,
)
from art_style_search.utils import ReasoningClient

logger = logging.getLogger(__name__)

_STRUCTURAL_CHANGE_TARGETS = ("caption_sections", "caption_length_target", "negative_prompt")


def _normalized_category_name(target_category: str, hypothesis: str, category_names: list[str]) -> str:
    return target_category if target_category else classify_hypothesis(hypothesis, category_names)


def _refinement_diversity_key(result: RefinementResult, category_names: list[str]) -> tuple[str, str, str]:
    return (
        _normalized_category_name(result.target_category, result.hypothesis, category_names),
        result.failure_mechanism.strip().lower(),
        result.intervention_type.strip().lower(),
    )


def _sketch_diversity_key(sketch: ExperimentSketch, category_names: list[str]) -> tuple[str, str, str]:
    return (
        _normalized_category_name(sketch.target_category, sketch.hypothesis, category_names),
        sketch.failure_mechanism.strip().lower(),
        sketch.intervention_type.strip().lower(),
    )


def _render_sketch(sketch: ExperimentSketch, idx: int) -> str:
    return (
        f"### Sketch {idx}\n"
        f"- hypothesis: {sketch.hypothesis}\n"
        f"- target_category: {sketch.target_category or '<infer>'}\n"
        f"- failure_mechanism: {sketch.failure_mechanism}\n"
        f"- intervention_type: {sketch.intervention_type}\n"
        f"- direction_id: {sketch.direction_id}\n"
        f"- direction_summary: {sketch.direction_summary}\n"
        f"- risk_level: {sketch.risk_level}\n"
        f"- expected_primary_metric: {sketch.expected_primary_metric}\n"
        f"- builds_on: {sketch.builds_on or 'none'}\n"
    )


def _experiment_system_prompt(*, current_template: PromptTemplate, response_format: str) -> str:
    section_name_list = ", ".join(section.name for section in current_template.sections)
    structural_change_targets = ", ".join(_STRUCTURAL_CHANGE_TARGETS)
    return (
        "You are an expert art director and prompt engineer optimizing a META-PROMPT.\n\n"
        "## TIER 1: NON-NEGOTIABLE RULES (CRITICAL)\n\n"
        "### Anchor sections (STRICT — never violate)\n"
        "1. The template MUST include 'style_foundation' as the FIRST section.\n"
        "   Why: this produces the [Art Style] block — shared style DNA that must be "
        "nearly IDENTICAL across all captions. A style_consistency metric measures this.\n"
        "2. The template MUST include 'subject_anchor' as the SECOND section.\n"
        "   Why: this produces the [Subject] block covering identity, features, pose, expression, "
        "and props — the single most important per-image block for reproduction fidelity.\n"
        "3. <caption_sections> MUST start with ['Art Style', 'Subject', ...].\n"
        "   Do not remove, rename, merge, or move these anchors. You MAY refine their content.\n\n"
        "### Knowledge Base constraints (CRITICAL)\n"
        "Read the KB Per-Category Status carefully.\n"
        "- 'Last rejected' entries are TESTED-AND-FAILED ideas. Do NOT repeat them. "
        "Why: repeating wastes a branch slot on a known failure.\n"
        "- Build on confirmed insights. Reference hypothesis IDs (e.g. 'builds on H3').\n"
        "- changed_section and changed_sections must use concrete template section names or "
        "structural change targets only. Never put taxonomy aliases (e.g. 'caption_structure') "
        "inside these fields. Why: these fields are diffed against the incumbent template to "
        "determine what changed — they must be matchable section names.\n\n"
        "### JSON format (STRICT)\n"
        "Return EXACTLY one JSON object. No markdown fences. No commentary before or after.\n\n"
        "### Convergence\n"
        "Only emit [CONVERGED] if ALL three hold: (1) deep plateau — multiple flat iterations, "
        "(2) every KB category has been targeted at least once, "
        "(3) you cannot name a concrete untried direction. "
        "The loop may reject [CONVERGED] — prefer a bold exploration branch over stopping.\n\n"
        "## TIER 2: HOW THE SYSTEM WORKS\n\n"
        "You are NOT writing image generation prompts directly. You are writing a META-PROMPT — "
        "instructions that tell Gemini Pro HOW to caption reference images. "
        "Those captions feed Gemini Flash to generate images. "
        "Pipeline: meta-prompt + reference -> caption -> generation -> compare with original.\n\n"
        "### Dual-purpose captions\n"
        "1. Recreate the reference image faithfully (measured by metrics).\n"
        "2. Embed REUSABLE art-style guidance in labeled sections that can later generate "
        "NEW art in the same style with different subjects.\n"
        "Embed core style rules as literal text the captioner weaves into every caption, "
        "plus per-image specific observations on top.\n\n"
        "### Meta-prompt requirements\n"
        "8-15 sections, each 4-8 sentences of instruction with embedded style rules. "
        "Total rendered prompt: 1200-1800 words. Cover: colors, technique, characters, "
        "background, composition, lighting, textures, mood, and what to AVOID.\n\n"
        "### Caption output structure\n"
        "Captions must have LABELED SECTIONS. First: [Art Style] (shared rules, identical across captions). "
        "Second: [Subject] (image-specific, 80-140 words). Remaining sections: your choice — "
        "that IS the optimization surface. Specify via <caption_sections> and <caption_length>.\n\n"
        "### Metric definitions\n"
        "Per-image metrics (generated vs paired original):\n"
        "- DreamSim (higher=better): perceptual similarity. 0.4=somewhat, 0.6=good, 0.8+=very close.\n"
        "- Color histogram (higher=better): palette match. 0.7=similar, 0.9+=very close.\n"
        "- SSIM (higher=better): structural similarity. 0.5=moderate, 0.7=good, 0.9+=near-identical.\n"
        "- HPS v2 (higher=better): caption-image alignment. Range 0.20-0.30.\n"
        "- Aesthetics (higher=better, 1-10): visual quality. 5=mediocre, 7=good, 8+=excellent.\n"
        "Vision scores (Gemini ternary: MATCH=1.0, PARTIAL=0.5, MISS=0.0):\n"
        "- vision_style: technique reproduction. vision_subject: subject fidelity. vision_composition: layout.\n"
        "Weights are ADAPTIVE — metrics with more variance across experiments get higher weight.\n\n"
        "## TIER 3: ITERATION STRATEGY\n\n"
        "Work in EXACTLY 3 directions (D1, D2, D3), ordered highest priority -> lowest.\n\n"
        "### Direction structure\n"
        "Each direction should include 1 targeted proposal + 1-3 bold proposals when brainstorming.\n"
        "- Targeted: change EXACTLY 1 section.\n"
        "- Bold: may change 1-3 related sections.\n"
        "Why this mix: targeted proposals give clean attribution; bold proposals test larger "
        "mechanisms that can leapfrog local optima.\n\n"
        "### What constitutes a bold proposal\n"
        "Must change information priority, scene-type policy, section schema, or a small cluster "
        "of related sections. Do NOT spend a bold slot on sentence counts or tiny wording polish.\n\n"
        "### Valid section names for this iteration\n"
        f"Concrete: {section_name_list}\n"
        f"Structural: {structural_change_targets}\n"
        "target_category may use taxonomy labels (e.g. 'caption_structure'). "
        "Multiple directions may touch the same category if they test DIFFERENT mechanisms.\n\n"
        "### Optimization dynamics\n"
        "**Momentum**: Double down on confirmed KB insights — explore if the same principle "
        "applies to other sections. Do not undo confirmed improvements unless metrics regressed.\n"
        "**Boldness**: Assume the incumbent is locally polished but conceptually wrong in at "
        "least one important way. Every direction must have at least one genuine bold variant.\n"
        "**Search depth**: Vary intervention type within a direction. Good types: information "
        "priority, negative constraints, scene-type split, schema change, multi-section rewrites.\n"
        "**Diversity**: Defined by (category, failure_mechanism, intervention_type). "
        "Do not emit two proposals sharing all three.\n\n"
        "## TIER 4: DIAGNOSTIC TIPS (use when relevant)\n\n"
        "- DreamSim weak -> captions miss structural/color/semantic details; make captioner more specific.\n"
        "- Per-image scores vary widely -> consider conditional captioning "
        "('for character images describe X; for backgrounds describe Y').\n"
        "- Use vision comparison and roundtrip feedback to identify what captions consistently miss.\n"
        "- Target the weakest KB category or build on partially confirmed hypotheses.\n"
        "- Use Open Problems to focus on highest-priority gaps.\n"
        "- Update <open_problems>: add new, remove solved, re-rank.\n\n"
        f"{response_format}"
    )


def _brainstorm_system(current_template: PromptTemplate, *, num_sketches: int) -> str:
    return _experiment_system_prompt(
        current_template=current_template,
        response_format=(
            "## EXECUTION CHECKLIST — verify before outputting\n"
            "- [ ] Every sketch has hypothesis, target_category, failure_mechanism, intervention_type, "
            "direction_id, direction_summary, risk_level, expected_primary_metric, and builds_on\n"
            "- [ ] builds_on is a single string (e.g. 'H3' or 'H3, H5') or an empty string, never null/array/object\n"
            "- [ ] 3 distinct direction_ids (D1, D2, D3) are present\n"
            "- [ ] The batch explores around two ideas per final branch slot\n\n"
            "Field types:\n"
            '- hypothesis, target_category, failure_mechanism, intervention_type, direction_id, direction_summary, risk_level, expected_primary_metric, builds_on: strings\n'
            f"Response format — one JSON object with 'sketches' array of length {num_sketches} "
            "and boolean 'converged'. Return JSON only. No markdown fences, no commentary."
        ),
    )


def _expand_system(current_template: PromptTemplate) -> str:
    return _experiment_system_prompt(
        current_template=current_template,
        response_format=(
            "## EXECUTION CHECKLIST — verify before outputting\n"
            "- [ ] The experiment has ALL required fields (analysis, lessons, hypothesis, builds_on, "
            "experiment, changed_section, changed_sections, target_category, direction_id, direction_summary, "
            "failure_mechanism, intervention_type, risk_level, expected_primary_metric, expected_tradeoff, "
            "open_problems, template_changes, template)\n"
            "- [ ] changed_sections[0] == changed_section\n"
            "- [ ] First template section is 'style_foundation', second is 'subject_anchor'\n"
            "- [ ] caption_sections starts with ['Art Style', 'Subject']\n"
            "- [ ] Total rendered template is 1200-1800 words\n\n"
            "Critical field types:\n"
            '- analysis: one string field, never an array\n'
            '- lessons: one JSON object with keys {"confirmed","rejected","new_insight"}, each a string\n'
            "- builds_on: a string like 'H3' or 'H3, H5', or an empty string\n\n"
            "Minimal wire-shape example:\n"
            '{"analysis":"...","lessons":{"confirmed":"","rejected":"","new_insight":"..."},"builds_on":"","hypothesis":"..."}\n\n'
            "Response format — one JSON object describing a single fully expanded experiment proposal. "
            "Return JSON only. No markdown fences, no commentary."
        ),
    )


def _rank_system() -> str:
    return (
        "You rank experiment sketches for expected impact on the art-style prompt search.\n\n"
        "Return JSON only. No markdown fences. No commentary.\n"
        "Rank sketches by expected improvement potential, novelty versus prior work, evidence fit with current "
        "feedback, and risk-reward balance. Prefer sketches that look both meaningful and executable.\n"
        "Output zero-based indices in best-to-worst order. Include every sketch at most once.\n"
        'Preferred exact wire shape: {"ranked_indices":[2,7,0,5]}'
    )


def _build_shared_proposal_user(
    style_profile: StyleProfile,
    current_template: PromptTemplate,
    knowledge_base: KnowledgeBase,
    best_metrics: AggregatedMetrics | None,
    last_results: list[IterationResult] | None,
    *,
    vision_feedback: str,
    roundtrip_feedback: str,
    caption_diffs: str,
) -> str:
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
        user_parts.append(f"\nCurrent composite score: {score:.4f}\n")

    kb_text = format_knowledge_base(knowledge_base)
    if kb_text:
        user_parts.append("\n\n")
        user_parts.append(kb_text)

    category_names = get_category_names(current_template)
    suggested = suggest_target_categories(knowledge_base, 3, category_names) if knowledge_base else []
    if suggested:
        user_parts.append(
            "\n## Suggested Target Categories (ranked by improvement potential; "
            "unexplored categories rank highest)\n" + "\n".join(f"{i}. {cat}" for i, cat in enumerate(suggested, 1))
        )

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
            worst = min(discarded, key=lambda r: composite_score(r.aggregated))
            worst_parts: list[str] = ["\n## Worst Experiment (learn from this failure)\n"]
            if worst.hypothesis:
                worst_parts.append(f"Hypothesis: {worst.hypothesis}\n")
            if worst.experiment:
                worst_parts.append(f"Experiment: {worst.experiment}\n")
            worst_parts.append(f"Metrics:\n{_format_metrics(worst.aggregated)}\n")
            if worst.per_image_scores and worst.iteration_captions:
                idx = min(range(len(worst.per_image_scores)), key=lambda i: worst.per_image_scores[i].dreamsim_similarity)
                if idx < len(worst.iteration_captions):
                    cap = worst.iteration_captions[idx]
                    cap_text = _truncate_words(cap.text, 150)
                    worst_parts.append(
                        f"Worst image ({cap.image_path.name}): "
                        f"DS={worst.per_image_scores[idx].dreamsim_similarity:.3f}\n"
                        f"Caption: {cap_text}\n"
                    )
            if worst.vision_feedback:
                worst_parts.append(f"Vision feedback: {_truncate_words(worst.vision_feedback, 100)}\n")
            user_parts.append("".join(worst_parts))

    if vision_feedback:
        user_parts.append("\n\n## Vision Comparison (Gemini analysis of generated vs reference images)\n")
        user_parts.append(_truncate_words(vision_feedback, 500, suffix="\n[...truncated]"))

    if roundtrip_feedback:
        user_parts.append("\n\n## Per-Image Results (sorted worst -> best by DreamSim)\n")
        user_parts.append(_truncate_words(roundtrip_feedback, 800, suffix="\n[...truncated]"))

    if caption_diffs:
        user_parts.append(f"\n\n{caption_diffs}")

    return "".join(user_parts)


def _brainstorm_user(
    shared_user: str,
    *,
    num_sketches: int,
    has_feedback: bool,
) -> str:
    instruction = (
        f"\n\nBrainstorm {num_sketches} lightweight experiment sketches in one JSON object. "
        "First diagnose likely failure mechanisms. Then produce sketches grouped into directions D1-D3, "
        "keeping each sketch lightweight and mechanism-specific. Do not emit full templates yet."
    )
    if has_feedback:
        instruction += " Use the vision comparison and per-image results to ground the sketches in concrete evidence."
    return shared_user + instruction


def _rank_user(
    sketches: list[ExperimentSketch],
    knowledge_base: KnowledgeBase,
    best_metrics: AggregatedMetrics | None,
) -> str:
    parts = ["## Candidate Sketches\n"]
    parts.extend(_render_sketch(sketch, idx) for idx, sketch in enumerate(sketches))
    kb_text = format_knowledge_base(knowledge_base, max_words=1000)
    if kb_text:
        parts.extend(["\n## Knowledge Base Summary\n", kb_text])
    if best_metrics:
        parts.extend(
            [
                "\n## Current Composite Score\n",
                f"{composite_score(best_metrics):.4f}\n",
                _format_metrics(best_metrics),
            ]
        )
    parts.append(
        f"\n\nValid indices are 0 through {len(sketches) - 1}.\n"
        'Return JSON only in the exact shape {"ranked_indices":[...]}.\n'
        "Favor likely winners without wasting slots on repeats."
    )
    return "".join(parts)


def _expand_user(shared_user: str, sketch: ExperimentSketch) -> str:
    return (
        f"{shared_user}\n\n## Sketch to Expand\n"
        f"{_render_sketch(sketch, 0)}\n"
        "Expand this sketch into one complete experiment proposal with a full template, lessons, change metadata, "
        "open problems, and expected tradeoff."
    )


def _stop_result(current_template: PromptTemplate) -> RefinementResult:
    return RefinementResult(
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


def _dedupe_sketches(sketches: list[ExperimentSketch], template: PromptTemplate) -> list[ExperimentSketch]:
    seen_keys: set[tuple[str, str, str]] = set()
    category_names = get_category_names(template)
    deduped: list[ExperimentSketch] = []
    for sketch in sketches:
        key = _sketch_diversity_key(sketch, category_names)
        if key in seen_keys:
            logger.warning(
                "Dropping duplicate sketch (category=%s, mechanism=%s, intervention=%s): %s",
                key[0],
                key[1] or "<none>",
                key[2] or "<none>",
                sketch.hypothesis[:80],
            )
            continue
        seen_keys.add(key)
        deduped.append(sketch)
    if len(deduped) < len(sketches):
        logger.info("Sketch diversity filter: kept %d/%d sketches", len(deduped), len(sketches))
    return deduped


def enforce_hypothesis_diversity(
    results: list[RefinementResult],
    template: PromptTemplate,
) -> list[RefinementResult]:
    """Deduplicate exact mechanism repeats while allowing multiple ideas per category."""
    seen_keys: set[tuple[str, str, str]] = set()
    diverse_results: list[RefinementResult] = []
    category_names = get_category_names(template)

    for r in results:
        key = _refinement_diversity_key(r, category_names)
        if key in seen_keys:
            logger.warning(
                "Dropping duplicate experiment (category=%s, mechanism=%s, intervention=%s): %s",
                key[0],
                key[1] or "<none>",
                key[2] or "<none>",
                r.hypothesis[:80],
            )
            continue
        seen_keys.add(key)
        diverse_results.append(r)

    if len(diverse_results) < len(results):
        logger.info("Diversity filter: kept %d/%d experiments", len(diverse_results), len(results))
    return diverse_results


def select_experiment_portfolio(
    results: list[RefinementResult],
    *,
    num_experiments: int,
    num_directions: int = 3,
) -> list[RefinementResult]:
    """Select a portfolio from raw proposals.

    Take one targeted proposal per direction first, preserving direction order,
    then fill remaining slots with bold proposals in original order.
    """
    if not results or num_experiments <= 0:
        return []

    direction_order: list[str] = []
    grouped: dict[str, list[RefinementResult]] = {}
    for idx, result in enumerate(results):
        direction_id = result.direction_id or f"UNGROUPED_{idx}"
        if direction_id not in grouped:
            direction_order.append(direction_id)
            grouped[direction_id] = []
        grouped[direction_id].append(result)

    selected: list[RefinementResult] = []
    seen_ids: set[int] = set()

    for direction_id in direction_order[:num_directions]:
        targeted = next((r for r in grouped[direction_id] if r.risk_level != "bold"), None)
        if targeted is None:
            continue
        selected.append(targeted)
        seen_ids.add(id(targeted))
        if len(selected) >= num_experiments:
            return selected[:num_experiments]

    bold_candidates = [r for r in results if r.risk_level == "bold" and id(r) not in seen_ids]
    for candidate in bold_candidates:
        selected.append(candidate)
        seen_ids.add(id(candidate))
        if len(selected) >= num_experiments:
            return selected[:num_experiments]

    for candidate in results:
        if id(candidate) in seen_ids:
            continue
        selected.append(candidate)
        seen_ids.add(id(candidate))
        if len(selected) >= num_experiments:
            break

    return selected[:num_experiments]


async def brainstorm_experiment_sketches(
    style_profile: StyleProfile,
    current_template: PromptTemplate,
    knowledge_base: KnowledgeBase,
    best_metrics: AggregatedMetrics | None,
    last_results: list[IterationResult] | None,
    *,
    client: ReasoningClient,
    model: str,
    num_sketches: int,
    vision_feedback: str = "",
    roundtrip_feedback: str = "",
    caption_diffs: str = "",
) -> tuple[list[ExperimentSketch], bool]:
    shared_user = _build_shared_proposal_user(
        style_profile,
        current_template,
        knowledge_base,
        best_metrics,
        last_results,
        vision_feedback=vision_feedback,
        roundtrip_feedback=roundtrip_feedback,
        caption_diffs=caption_diffs,
    )
    user = _brainstorm_user(shared_user, num_sketches=num_sketches, has_feedback=bool(vision_feedback or roundtrip_feedback))
    logger.info("Brainstorming %d experiment sketches (%s) — context: ~%d words", num_sketches, model, len(user.split()))
    sketches, converged = await client.call_json(
        model=model,
        system=_brainstorm_system(current_template, num_sketches=num_sketches),
        user=user,
        validator=lambda data: validate_brainstorm_payload(data, num_sketches=num_sketches),
        response_name="brainstorm",
        schema_hint=schema_hint("brainstorm"),
        max_tokens=16000,
        repair_retries=2,
    )
    if not sketches:
        logger.warning("No valid sketches parsed from brainstorm response")
    return sketches, converged


async def rank_experiment_sketches(
    sketches: list[ExperimentSketch],
    knowledge_base: KnowledgeBase,
    best_metrics: AggregatedMetrics | None,
    *,
    client: ReasoningClient,
    model: str,
) -> list[ExperimentSketch]:
    if len(sketches) <= 1:
        return list(sketches)
    try:
        ranked_indices = await client.call_json(
            model=model,
            system=_rank_system(),
            user=_rank_user(sketches, knowledge_base, best_metrics),
            validator=lambda data: validate_ranking_payload(data, num_sketches=len(sketches)),
            response_name="ranking",
            schema_hint=schema_hint("ranking"),
            max_tokens=1000,
            repair_retries=1,
            final_failure_log_level=logging.INFO,
        )
    except Exception as exc:
        logger.info("Ranking failed; falling back to brainstorm order: %s: %s", type(exc).__name__, exc)
        return list(sketches)
    return [sketches[idx] for idx in ranked_indices]


async def expand_experiment_sketches(
    style_profile: StyleProfile,
    current_template: PromptTemplate,
    knowledge_base: KnowledgeBase,
    best_metrics: AggregatedMetrics | None,
    last_results: list[IterationResult] | None,
    *,
    client: ReasoningClient,
    model: str,
    sketches: list[ExperimentSketch],
    vision_feedback: str = "",
    roundtrip_feedback: str = "",
    caption_diffs: str = "",
) -> list[RefinementResult]:
    if not sketches:
        return []

    shared_user = _build_shared_proposal_user(
        style_profile,
        current_template,
        knowledge_base,
        best_metrics,
        last_results,
        vision_feedback=vision_feedback,
        roundtrip_feedback=roundtrip_feedback,
        caption_diffs=caption_diffs,
    )
    system = _expand_system(current_template)
    tasks = [
        client.call_json(
            model=model,
            system=system,
            user=_expand_user(shared_user, sketch),
            validator=validate_expansion_payload,
            response_name=f"expansion_{idx}",
            schema_hint=schema_hint("expansion"),
            max_tokens=16000,
            repair_retries=1,
        )
        for idx, sketch in enumerate(sketches)
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[RefinementResult] = []
    failures = 0
    for raw in raw_results:
        if isinstance(raw, BaseException):
            failures += 1
            logger.warning("Expansion failed: %s: %s", type(raw).__name__, raw)
            continue
        results.append(raw)
    logger.info("Expansion finished: %d/%d succeeded", len(results), len(sketches))
    if failures and not results:
        logger.warning("All experiment expansions failed")
    return results


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
    """Compatibility wrapper: brainstorm more ideas, rank them, then expand survivors."""

    num_sketches = max(num_experiments * 2, num_experiments)
    sketches, converged = await brainstorm_experiment_sketches(
        style_profile,
        current_template,
        knowledge_base,
        best_metrics,
        last_results,
        client=client,
        model=model,
        num_sketches=num_sketches,
        vision_feedback=vision_feedback,
        roundtrip_feedback=roundtrip_feedback,
        caption_diffs=caption_diffs,
    )
    if not sketches:
        return [_stop_result(current_template)] if converged else []

    ranked = await rank_experiment_sketches(
        sketches,
        knowledge_base,
        best_metrics,
        client=client,
        model=model,
    )
    ranked = _dedupe_sketches(ranked[:num_experiments], current_template)
    results = await expand_experiment_sketches(
        style_profile,
        current_template,
        knowledge_base,
        best_metrics,
        last_results,
        client=client,
        model=model,
        sketches=ranked,
        vision_feedback=vision_feedback,
        roundtrip_feedback=roundtrip_feedback,
        caption_diffs=caption_diffs,
    )

    if converged:
        if results:
            results[-1].should_stop = True
        else:
            results.append(_stop_result(current_template))
    return results


__all__ = [
    "brainstorm_experiment_sketches",
    "enforce_hypothesis_diversity",
    "expand_experiment_sketches",
    "propose_experiments",
    "rank_experiment_sketches",
    "select_experiment_portfolio",
]

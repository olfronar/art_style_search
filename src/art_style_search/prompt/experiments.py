"""Per-iteration experiment proposals: propose raw candidates, then build a direction portfolio."""

from __future__ import annotations

import logging

from art_style_search.contracts import Lessons, RefinementResult
from art_style_search.prompt._format import (
    _format_metrics,
    _format_style_profile,
    _format_template,
    _truncate_words,
    format_knowledge_base,
    suggest_target_categories,
)
from art_style_search.prompt.json_contracts import schema_hint, validate_experiment_batch_payload
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


def _normalized_category(result: RefinementResult, category_names: list[str]) -> str:
    return result.target_category if result.target_category else classify_hypothesis(result.hypothesis, category_names)


def _diversity_key(result: RefinementResult, category_names: list[str]) -> tuple[str, str, str]:
    return (
        _normalized_category(result, category_names),
        result.failure_mechanism.strip().lower(),
        result.intervention_type.strip().lower(),
    )


def enforce_hypothesis_diversity(
    results: list[RefinementResult],
    template: PromptTemplate,
) -> list[RefinementResult]:
    """Deduplicate exact mechanism repeats while allowing multiple ideas per category."""
    seen_keys: set[tuple[str, str, str]] = set()
    diverse_results: list[RefinementResult] = []
    category_names = get_category_names(template)

    for r in results:
        key = _diversity_key(r, category_names)
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
    """Propose a raw batch of experiments in a single reasoning-model call."""

    section_name_list = ", ".join(section.name for section in current_template.sections)
    structural_change_targets = ", ".join(["caption_sections", "caption_length_target", "negative_prompt"])
    system = (
        "You are an expert art director and prompt engineer optimizing a META-PROMPT.\n\n"
        # ── TIER 1: NON-NEGOTIABLE (CRITICAL) ──────────────────────────────
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
        # ── TIER 2: CORE TASK (MANDATORY) ──────────────────────────────────
        "## TIER 2: HOW THE SYSTEM WORKS\n\n"
        "You are NOT writing image generation prompts directly. You are writing a META-PROMPT — "
        "instructions that tell Gemini Pro HOW to caption reference images. "
        "Those captions feed Gemini Flash to generate images. "
        "Pipeline: meta-prompt + reference → caption → generation → compare with original.\n\n"
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
        # ── TIER 3: EXECUTION DETAILS ──────────────────────────────────────
        "## TIER 3: ITERATION STRATEGY\n\n"
        f"Propose exactly {num_experiments} experiments in EXACTLY 3 directions (D1, D2, D3), "
        "ordered highest priority → lowest.\n\n"
        "### Direction structure\n"
        "Each direction: 1 targeted proposal + 1-3 bold proposals.\n"
        "- Targeted: change EXACTLY 1 section. Appears first for that direction.\n"
        "- Bold: may change 1-3 related sections. Appears after targeted, strongest → weakest.\n"
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
        # ── TIER 4: DIAGNOSTIC TIPS ───────────────────────────────────────
        "## TIER 4: DIAGNOSTIC TIPS (use when relevant)\n\n"
        "- DreamSim weak → captions miss structural/color/semantic details; make captioner more specific.\n"
        "- Per-image scores vary widely → consider conditional captioning "
        "('for character images describe X; for backgrounds describe Y').\n"
        "- Use vision comparison and roundtrip feedback to identify what captions consistently miss.\n"
        "- Target the weakest KB category or build on partially confirmed hypotheses.\n"
        "- Use Open Problems to focus on highest-priority gaps.\n"
        "- Update <open_problems>: add new, remove solved, re-rank.\n\n"
        # ── EXECUTION CHECKLIST ─────────────────────────────────────────────
        "## EXECUTION CHECKLIST — verify before outputting\n"
        "- [ ] Every experiment has ALL 16 required fields (analysis, lessons, hypothesis, builds_on, "
        "experiment, changed_section, changed_sections, target_category, direction_id, direction_summary, "
        "failure_mechanism, intervention_type, risk_level, expected_primary_metric, expected_tradeoff, "
        "open_problems, template_changes, template)\n"
        "- [ ] changed_sections[0] == changed_section in every experiment\n"
        "- [ ] 3 distinct direction_ids (D1, D2, D3), each with exactly 1 targeted first\n"
        "- [ ] First template section is 'style_foundation', second is 'subject_anchor'\n"
        "- [ ] caption_sections starts with ['Art Style', 'Subject']\n"
        "- [ ] No rejected KB mechanism/intervention pair is repeated\n"
        "- [ ] Total rendered template is 1200-1800 words\n\n"
        f"Response format — one JSON object with 'experiments' array of length {num_experiments} "
        "and boolean 'converged'. Return JSON only. No markdown fences, no commentary."
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
        user_parts.append(f"\nCurrent composite score: {score:.4f}\n")

    # Knowledge base — structured lessons from all previous experiments
    kb_text = format_knowledge_base(knowledge_base)
    if kb_text:
        user_parts.append("\n\n")
        user_parts.append(kb_text)

    # Suggest target categories for diversity — always render when a ranking is
    # available, including on iteration 1, so the unexplored synonym-map categories
    # (lighting, texture, background, caption_structure, …) surface at priority 1.0.
    category_names = get_category_names(current_template)
    suggested = suggest_target_categories(knowledge_base, 3, category_names) if knowledge_base else []
    if suggested:
        user_parts.append(
            "\n## Suggested Target Categories (ranked by improvement potential; "
            "unexplored categories rank highest)\n" + "\n".join(f"{i}. {cat}" for i, cat in enumerate(suggested, 1))
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
        # Cap vision feedback to ~500 words to prevent context degradation
        user_parts.append(_truncate_words(vision_feedback, 500, suffix="\n[...truncated]"))

    if roundtrip_feedback:
        user_parts.append("\n\n## Per-Image Results (sorted worst → best by DreamSim)\n")
        # Cap roundtrip feedback to ~800 words (full detail for worst images, metrics-only for rest)
        user_parts.append(_truncate_words(roundtrip_feedback, 800, suffix="\n[...truncated]"))

    if caption_diffs:
        user_parts.append(f"\n\n{caption_diffs}")

    has_feedback = vision_feedback or roundtrip_feedback
    instruction = (
        f"\n\nPropose {num_experiments} improved templates in one JSON object. "
        "First diagnose 3 failure mechanisms. Then produce 3 directions (D1-D3) with one targeted proposal "
        "plus bold variants inside each direction. Use the Knowledge Base to avoid repeating the same mechanism/intervention pair. "
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

    results, converged = await client.call_json(
        model=model,
        system=system,
        user=user,
        validator=lambda data: validate_experiment_batch_payload(data, num_experiments=num_experiments),
        response_name="experiment_batch",
        schema_hint=schema_hint("experiment_batch"),
        max_tokens=30000,
        repair_retries=2,
    )

    if converged:
        if results:
            results[-1].should_stop = True
        else:
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

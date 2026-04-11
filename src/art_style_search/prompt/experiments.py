"""Per-iteration experiment proposals: propose N branches in one call, then dedup."""

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


def enforce_hypothesis_diversity(
    results: list[RefinementResult],
    template: PromptTemplate,
) -> list[RefinementResult]:
    """Deduplicate experiments targeting the same category. Keep the first occurrence."""
    seen_categories: set[str] = set()
    diverse_results: list[RefinementResult] = []
    category_names = get_category_names(template)

    for r in results:
        # Prefer the model's explicit target_category; fall back to keyword classification
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
    """Propose N experiments in a single reasoning-model call.

    Uses ``<branch>`` tags so the model generates all experiments at once,
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
        "- SSIM (higher=better): pixel-level structural similarity. 0.5=moderate, 0.7=good, 0.9+=near-identical.\n"
        "- HPS v2 (higher=better): caption-image alignment. Range 0.20-0.30.\n"
        "- Aesthetics (higher=better, 1-10): visual quality. 5=mediocre, 7=good, 8+=excellent.\n"
        "Per-image vision scores (from Gemini visual comparison, ternary: MATCH=1.0, PARTIAL=0.5, MISS=0.0):\n"
        "- vision_style: art technique reproduction (aggregated as ratio of images matching).\n"
        "- vision_subject: character/subject fidelity.\n"
        "- vision_composition: spatial layout accuracy.\n"
        "Weights are ADAPTIVE — metrics with more variance across experiments get higher weight.\n\n"
        "## Iteration strategy\n"
        f"- Propose exactly {num_experiments} experiments in a single JSON response. "
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
        "Only emit [CONVERGED] (at the very end of the LAST branch) if you have verified "
        "ALL of the following: (1) the plateau is deep — multiple flat iterations in a row, "
        "(2) every hypothesis category in the knowledge base has been directly targeted at "
        "least once, and (3) you cannot name a concrete untried direction. The orchestration "
        "loop reserves the right to reject [CONVERGED] and request a new batch if these "
        "conditions are not met — prefer proposing a bold exploration branch over stopping.\n\n"
        f"Response format — return EXACTLY one JSON object with an 'experiments' array of length {num_experiments} "
        "and a top-level boolean 'converged'. Each experiment object must contain:\n"
        "- analysis\n"
        "- lessons: {confirmed, rejected, new_insight}\n"
        "- hypothesis\n"
        "- builds_on (string or null)\n"
        "- experiment\n"
        "- changed_section\n"
        "- target_category\n"
        "- open_problems (array of strings)\n"
        "- template_changes\n"
        "- template: {sections, negative_prompt, caption_sections, caption_length_target}\n"
        "Return JSON only. No markdown fences, no commentary."
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
    kb_text = format_knowledge_base(knowledge_base)
    if kb_text:
        user_parts.append("\n\n")
        user_parts.append(kb_text)

    # Suggest target categories for diversity — always render when a ranking is
    # available, including on iteration 1, so the unexplored synonym-map categories
    # (lighting, texture, background, caption_structure, …) surface at priority 1.0.
    category_names = get_category_names(current_template)
    suggested = suggest_target_categories(knowledge_base, num_experiments, category_names) if knowledge_base else []
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

    results, converged = await client.call_json(
        model=model,
        system=system,
        user=user,
        validator=lambda data: validate_experiment_batch_payload(data, num_experiments=num_experiments),
        response_name="experiment_batch",
        schema_hint=schema_hint("experiment_batch"),
        max_tokens=30000,
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

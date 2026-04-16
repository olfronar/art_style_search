"""Per-iteration experiment proposals: brainstorm sketches, rank them, then expand."""

from __future__ import annotations

import asyncio
import logging

from art_style_search.caption_sections import parse_labeled_sections
from art_style_search.contracts import ExperimentSketch, Lessons, RefinementResult
from art_style_search.prompt._format import (
    _format_metrics,
    _format_style_profile,
    _format_template,
    _truncate_words,
    format_knowledge_base,
    suggest_target_categories,
)
from art_style_search.prompt._parse import _MAX_RENDERED_WORDS, _MIN_RENDERED_WORDS
from art_style_search.prompt.json_contracts import (
    response_schema,
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


def _caption_feedback_excerpt(text: str, max_words: int) -> str:
    """Prioritize the most informative labeled sections before truncating."""
    parsed = parse_labeled_sections(text)
    if not parsed:
        return _truncate_words(text, max_words)

    parts: list[str] = []
    seen: set[str] = set()
    for name in ("Subject", "Art Style"):
        body = parsed.get(name, "").strip()
        if body:
            parts.append(f"[{name}] {body}")
            seen.add(name)
    for name, body in parsed.items():
        stripped = body.strip()
        if name in seen or not stripped:
            continue
        parts.append(f"[{name}] {stripped}")

    if not parts:
        return _truncate_words(text, max_words)
    return _truncate_words("\n\n".join(parts), max_words)


_BRAINSTORM_EXAMPLE = (
    "## Example of a good sketch\n"
    '{"hypothesis":"Subject descriptions use generic terms (person, figure) instead of specific identity cues, '
    'causing the generator to produce wrong subjects","target_category":"subject_anchor",'
    '"failure_mechanism":"Identity cues buried behind style language — the generator attends to style tokens first and loses subject specificity",'
    '"intervention_type":"information_priority","direction_id":"D1","direction_summary":"Subject identity lock",'
    '"risk_level":"targeted","expected_primary_metric":"vision_subject","builds_on":"H3"}\n\n'
    "## Example of a bad sketch (too vague — avoid this)\n"
    '{"hypothesis":"Make captions better","target_category":"","failure_mechanism":"Captions are not good enough",'
    '"intervention_type":"general_improvement","direction_id":"D1","direction_summary":"Improve things",'
    '"risk_level":"targeted","expected_primary_metric":"all","builds_on":""}'
)


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


def _experiment_system_prompt(
    *, current_template: PromptTemplate, response_format: str, is_first_iteration: bool = False
) -> str:
    section_name_list = ", ".join(section.name for section in current_template.sections)
    structural_change_targets = ", ".join(_STRUCTURAL_CHANGE_TARGETS)

    # Base rules — always present
    base = (
        "You are an expert art director optimizing a META-PROMPT — instructions telling Gemini Pro how to caption reference images.\n\n"
        "## NON-NEGOTIABLE RULES\n\n"
        "### Anchor sections (STRICT — never violate)\n"
        "1. First section MUST be 'style_foundation' — produces [Art Style] block (shared style DNA, measured by style_consistency).\n"
        "2. Second section MUST be 'subject_anchor' — produces [Subject] block (identity, features, pose — most important for reproduction).\n"
        "3. caption_sections MUST start with ['Art Style', 'Subject', ...]. Never remove/rename these anchors.\n"
        "4. style_foundation.value MUST contain a 'How to Draw:' sub-block naming silhouette primitives, "
        "construction order, line policy, shading layers (base -> AO -> midtones -> rim -> specular), edge softness, "
        "texture grain, and signature rendering quirk. Refinements may reword this sub-block but MUST keep it present.\n"
        "5. subject_anchor.value MUST contain a 'Proportions:' sub-block requiring numeric head-heights-tall AND "
        "an archetype token (chibi / stylized-youth / heroic / realistic-adult / elongated).\n"
        "6. style_foundation MUST name the observed medium class (one of A hand-drawn 2D / B vector-flat 2D / "
        "C stylized 3D CGI / D photoreal 3D / E mixed-2.5D) and direct the captioner to use class-appropriate "
        "vocabulary only. Swapping medium class is a legitimate bold experiment; silently flipping vocabulary "
        "without flipping the declared class is not.\n"
        "7. LACONIC: state each style rule ONCE in style_foundation. Other sections reference rules by name. "
        "Pure verbatim restatement across sections is a regression.\n\n"
        "### Knowledge Base constraints\n"
        "- 'Last rejected' entries are TESTED-AND-FAILED — do NOT repeat them.\n"
        "- Build on confirmed insights (reference hypothesis IDs like 'builds on H3').\n"
        "- changed_section/changed_sections must use concrete template section names, not taxonomy aliases.\n\n"
        "### Output format\n"
        "Return EXACTLY one JSON object. No markdown fences. No commentary.\n\n"
        "### Convergence\n"
        "Only emit [CONVERGED] if ALL three hold: (1) deep plateau, (2) every KB category targeted, (3) no untried direction. "
        "Prefer a bold exploration branch over stopping.\n\n"
    )

    if is_first_iteration:
        context = (
            "## SYSTEM CONTEXT\n\n"
            "Pipeline: meta-prompt + reference -> Gemini Pro caption -> Gemini Flash generation -> compare with original.\n"
            "Captions serve TWO purposes: (1) recreate the image faithfully, (2) embed REUSABLE art-style guidance in labeled sections.\n"
            "style_foundation is the shared DNA the captioner repeats inside every caption's [Art Style] block; ancillary sections "
            "layer per-image observations on top without restating the shared rules.\n\n"
            "### Meta-prompt structure\n"
            "8-20 sections, 2000-8000 words total. Cover: colors, technique, characters, background, composition, lighting, textures, mood, and what to AVOID.\n\n"
            "### Caption output structure\n"
            "First: [Art Style] (shared rules, identical across captions, often 1000-2000 words). "
            "Second: [Subject] (image-specific and most important, often 1000-2000 words). "
            "Remaining sections typically run 150-400 words each — they are your optimization surface.\n\n"
            "### Metrics\n"
            "| Metric | Range | Good | Description |\n"
            "|--------|-------|------|-------------|\n"
            "| DreamSim | 0-1 | 0.6+ | Perceptual similarity |\n"
            "| Color histogram | 0-1 | 0.7+ | Palette match |\n"
            "| SSIM | 0-1 | 0.7+ | Structural similarity |\n"
            "| HPS v2 | 0.2-0.3 | >0.25 | Caption-image alignment |\n"
            "| Aesthetics | 1-10 | 7+ | Visual quality |\n"
            "| vision_style/subject/composition | 0-1 | MATCH=1.0, PARTIAL=0.5, MISS=0.0 | Gemini ternary |\n"
            "Weights are ADAPTIVE — metrics with more variance across experiments get higher weight.\n\n"
        )
    else:
        context = (
            "## Metrics reminder\n"
            "Same pipeline and metrics as before: DreamSim, Color histogram, SSIM, HPS v2, Aesthetics, Vision (style/subject/composition).\n\n"
        )

    strategy = (
        "## ITERATION STRATEGY\n\n"
        "Work in EXACTLY 3 directions (D1 > D2 > D3 by priority).\n"
        "Each direction contains 1 targeted proposal and 1 to 3 bold proposals.\n"
        "Section-count rules (HARD CAP — proposals that violate these are rejected):\n"
        "- targeted proposal: changed_sections has exactly 1 entry.\n"
        "- bold proposal: changed_sections has 1, 2, or 3 entries. Never 4 or more.\n"
        "Bold proposals must change information priority, scene-type policy, section schema, or a cluster of related sections — not wording polish.\n\n"
        "### Valid section names\n"
        f"Concrete: {section_name_list}\n"
        f"Structural: {structural_change_targets}\n"
        "target_category may use taxonomy labels. Multiple directions may touch the same category with DIFFERENT mechanisms.\n\n"
        "### Structural exploration quota (HARD RULE)\n"
        "At least 1 proposal in the batch MUST change template STRUCTURE, not just section wording. "
        "A 'structural change' means AT LEAST ONE of:\n"
        "- modifying caption_sections (add, remove, or reorder labeled output sections), OR\n"
        "- changing caption_length_target by >=30% from the incumbent, OR\n"
        "- adding, removing, or renaming an entry in the template's sections list (the section_names set).\n"
        "Rewording an existing section's value — however extensively — does NOT count as structural. "
        "Without structural exploration the search converges to one schema and the loop stops learning. "
        "If the incumbent's (section_names, caption_sections, caption_length_target) triple has held steady for 2+ iterations, at least one bold proposal MUST change AT LEAST TWO of those three.\n\n"
        "### Search principles\n"
        "- **Momentum**: Double down on confirmed KB insights. Do not undo confirmed improvements.\n"
        "- **Boldness**: Assume the incumbent is conceptually wrong in at least one way. Every direction needs a bold variant.\n"
        "- **Diversity**: No two proposals may share (category, failure_mechanism, intervention_type).\n"
        "- **Structural novelty**: Pure value-rewording batches are the most common failure mode of this optimizer. Explicitly consider section-schema or length changes alongside content edits.\n\n"
        "## DIAGNOSTIC TIPS\n"
        "- DreamSim weak -> captions miss structural/semantic details.\n"
        "- Per-image scores vary -> consider conditional captioning.\n"
        "- Use vision/roundtrip feedback to find what captions miss.\n"
        "- Target weakest KB category or build on partially confirmed hypotheses.\n\n"
        f"{response_format}"
    )

    return base + context + strategy


def _brainstorm_system(current_template: PromptTemplate, *, num_sketches: int, is_first_iteration: bool = False) -> str:
    return _experiment_system_prompt(
        current_template=current_template,
        is_first_iteration=is_first_iteration,
        response_format=(
            f"{_BRAINSTORM_EXAMPLE}\n\n"
            "## EXECUTION CHECKLIST — verify before outputting\n"
            "- [ ] Every sketch has hypothesis, target_category, failure_mechanism, intervention_type, "
            "direction_id, direction_summary, risk_level, expected_primary_metric, and builds_on\n"
            "- [ ] builds_on is a single string (e.g. 'H3' or 'H3, H5') or an empty string, never null/array/object\n"
            "- [ ] 3 distinct direction_ids (D1, D2, D3) are present\n"
            "- [ ] The batch explores around two ideas per final branch slot\n\n"
            "Field types:\n"
            "- hypothesis, target_category, failure_mechanism, intervention_type, direction_id, direction_summary, risk_level, expected_primary_metric, builds_on: strings\n"
            f"Response format — one JSON object with 'sketches' array of length {num_sketches} "
            "and boolean 'converged'. Return JSON only. No markdown fences, no commentary."
        ),
    )


def _expand_system(current_template: PromptTemplate, *, is_first_iteration: bool = False) -> str:
    return _experiment_system_prompt(
        current_template=current_template,
        is_first_iteration=is_first_iteration,
        response_format=(
            "## EXECUTION CHECKLIST — verify before outputting\n"
            "- [ ] The experiment has ALL required fields (analysis, lessons, hypothesis, builds_on, "
            "experiment, changed_section, changed_sections, target_category, direction_id, direction_summary, "
            "failure_mechanism, intervention_type, risk_level, expected_primary_metric, expected_tradeoff, "
            "open_problems, template_changes, template)\n"
            "- [ ] changed_sections[0] == changed_section\n"
            "- [ ] First template section is 'style_foundation', second is 'subject_anchor'\n"
            "- [ ] caption_sections starts with ['Art Style', 'Subject']\n"
            "- [ ] Total rendered template is 2000-8000 words\n"
            "- [ ] style_foundation.value contains a 'How to Draw:' sub-block (never silently dropped during a refinement)\n"
            "- [ ] subject_anchor.value contains a 'Proportions:' sub-block + at least one archetype token "
            "(chibi / stylized-youth / heroic / realistic-adult / elongated)\n"
            "- [ ] style_foundation names the medium class (A/B/C/D/E) and uses class-appropriate vocabulary\n"
            "- [ ] Each style rule is stated ONCE — no verbatim duplication across sections\n\n"
            "Critical field types:\n"
            "- analysis: one string field, never an array\n"
            '- lessons: one JSON object with keys {"confirmed","rejected","new_insight"}, each a string\n'
            "- builds_on: a string like 'H3' or 'H3, H5', or an empty string\n\n"
            "- template: one JSON object with keys sections, negative_prompt, caption_sections, caption_length_target "
            "(never XML text)\n"
            "- template_changes: one summary string, never an object\n"
            "- open_problems: a JSON array of strings, even when there is only one item\n\n"
            "Minimal wire-shape example:\n"
            '{"analysis":"...","lessons":{"confirmed":"","rejected":"","new_insight":"..."},"builds_on":"","hypothesis":"..."}\n\n'
            "Response format — one JSON object describing a single fully expanded experiment proposal. "
            "Return JSON only. No markdown fences, no commentary."
        ),
    )


def _rank_system() -> str:
    return (
        "You rank experiment sketches for expected impact on the art-style prompt search.\n\n"
        "Rank by these criteria in priority order:\n"
        "1. **Evidence fit** (primary): Does the sketch directly address observed failures from vision/roundtrip feedback?\n"
        "2. **Risk-reward balance** (secondary): Targeted sketches that address real failures beat bold but ungrounded ones.\n"
        "3. **Novelty** (tertiary): Prefer exploring new mechanisms over repeating similar ones.\n\n"
        "A sketch is 'executable' when its failure_mechanism is specific enough to map to a concrete template change.\n"
        "Prefer sketches that look both meaningful and executable.\n\n"
        "Return JSON only. No markdown fences. No commentary.\n"
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
    vision_feedback_word_limit: int = 300,
    roundtrip_feedback_word_limit: int = 400,
    include_roundtrip_feedback: bool = True,
    iteration: int = 0,
    plateau_counter: int = 0,
) -> str:
    has_history = knowledge_base.hypotheses
    user_parts: list[str] = []

    # Skip Style Profile after iteration 1 — it never changes
    if iteration < 2:
        user_parts.append("## Style Profile\n")
        user_parts.append(_format_style_profile(style_profile, compact=bool(has_history)))

    # Current Template: after iteration 1, show rendered text only (drop XML format)
    user_parts.append("\n\n## Current Template\n")
    if iteration < 2:
        user_parts.append(_format_template(current_template))
        user_parts.append(f"\nRendered prompt: {current_template.render()}")
    else:
        user_parts.append(current_template.render())

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
                idx = min(
                    range(len(worst.per_image_scores)), key=lambda i: worst.per_image_scores[i].dreamsim_similarity
                )
                if idx < len(worst.iteration_captions):
                    cap = worst.iteration_captions[idx]
                    cap_text = _caption_feedback_excerpt(cap.text, 800)
                    worst_parts.append(
                        f"Worst image ({cap.image_path.name}): "
                        f"DS={worst.per_image_scores[idx].dreamsim_similarity:.3f}\n"
                        f"Caption: {cap_text}\n"
                    )
            if worst.vision_feedback:
                worst_parts.append(f"Vision feedback: {_truncate_words(worst.vision_feedback, 400)}\n")
            user_parts.append("".join(worst_parts))

    # Adaptive vision feedback word limit based on plateau depth
    effective_vision_limit = vision_feedback_word_limit
    if plateau_counter > 3:
        effective_vision_limit = 500

    if vision_feedback:
        user_parts.append("\n\n## Vision Comparison (Gemini analysis of generated vs reference images)\n")
        user_parts.append(_truncate_words(vision_feedback, effective_vision_limit, suffix="\n[...truncated]"))

    if roundtrip_feedback and include_roundtrip_feedback:
        user_parts.append("\n\n## Per-Image Results (sorted worst -> best by DreamSim)\n")
        user_parts.append(_truncate_words(roundtrip_feedback, roundtrip_feedback_word_limit, suffix="\n[...truncated]"))

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
        "keeping each sketch lightweight and mechanism-specific. Use the failure_mechanism field to record that diagnosis. "
        "Do not emit full templates yet."
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
    kb_text = format_knowledge_base(knowledge_base, max_words=2000)
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


_CAPTION_ANCHOR_MIN_WORDS = 100  # [Art Style] / [Subject] floor (matches caption.py validation)
_CAPTION_ANCILLARY_MIN_WORDS = 80  # Other labeled caption sections


def _minimum_plausible_caption_total(caption_sections: list[str]) -> int:
    """Minimum caption length implied by anchor + ancillary section floors."""
    n = len(caption_sections)
    if n == 0:
        return 0
    n_anchors = min(n, 2)
    n_ancillary = max(n - 2, 0)
    return n_anchors * _CAPTION_ANCHOR_MIN_WORDS + n_ancillary * _CAPTION_ANCILLARY_MIN_WORDS


def _validate_expanded_template(result: RefinementResult, incumbent: PromptTemplate) -> list[str]:
    """Programmatic post-expand verification. Returns list of violation descriptions."""
    _ = incumbent  # reserved for future diff-based validations
    issues: list[str] = []
    tmpl = result.template

    # Check anchor positions
    if tmpl.sections and tmpl.sections[0].name != "style_foundation":
        issues.append(f"First section is '{tmpl.sections[0].name}', expected 'style_foundation'")
    if len(tmpl.sections) > 1 and tmpl.sections[1].name != "subject_anchor":
        issues.append(f"Second section is '{tmpl.sections[1].name}', expected 'subject_anchor'")

    # Check word count
    rendered = tmpl.render()
    word_count = len(rendered.split())
    if word_count < _MIN_RENDERED_WORDS or word_count > _MAX_RENDERED_WORDS:
        issues.append(f"Rendered template is {word_count} words (target: {_MIN_RENDERED_WORDS}-{_MAX_RENDERED_WORDS})")

    # Check changed_sections consistency
    if result.changed_sections and result.changed_section and result.changed_sections[0] != result.changed_section:
        issues.append(
            f"changed_sections[0]='{result.changed_sections[0]}' != changed_section='{result.changed_section}'"
        )

    # N3: Self-contradictory caption_length_target — the reasoner asks for fewer total words than
    # its own caption_sections floor can produce. Captioner will always overshoot such targets.
    if tmpl.caption_length_target > 0 and tmpl.caption_sections:
        min_plausible_total = _minimum_plausible_caption_total(tmpl.caption_sections)
        if tmpl.caption_length_target < min_plausible_total:
            issues.append(
                f"caption_length_target={tmpl.caption_length_target} is below minimum plausible total "
                f"({min_plausible_total}w) for {len(tmpl.caption_sections)} caption sections "
                f"(floor = 2*{_CAPTION_ANCHOR_MIN_WORDS} + {max(len(tmpl.caption_sections) - 2, 0)}*{_CAPTION_ANCILLARY_MIN_WORDS})"
            )

    return issues


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


def template_structural_signature(template: PromptTemplate) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    """Return a hashable signature capturing a template's structural identity.

    Two templates share a signature when their section_names, caption_sections order,
    and caption_length_target are identical — i.e., only section VALUES may differ.
    """
    section_names = tuple(section.name for section in template.sections)
    caption_sections = tuple(template.caption_sections)
    return (section_names, caption_sections, template.caption_length_target)


def _length_target_delta_ratio(a: int, b: int) -> float:
    if max(a, b) <= 0:
        return 0.0
    return abs(a - b) / max(a, b)


def _is_structurally_novel(
    candidate: PromptTemplate,
    incumbent: PromptTemplate,
    *,
    length_delta_threshold: float = 0.30,
) -> bool:
    """True when candidate differs from incumbent on at least one structural axis.

    Structural axes: section_names set/order, caption_sections order, caption_length_target (>=threshold delta).
    Value-only edits return False.
    """
    cand_sig = template_structural_signature(candidate)
    incumb_sig = template_structural_signature(incumbent)
    if cand_sig[0] != incumb_sig[0]:
        return True
    if cand_sig[1] != incumb_sig[1]:
        return True
    return _length_target_delta_ratio(cand_sig[2], incumb_sig[2]) >= length_delta_threshold


def select_experiment_portfolio(
    results: list[RefinementResult],
    *,
    num_experiments: int,
    num_directions: int = 3,
    incumbent_template: PromptTemplate | None = None,
) -> list[RefinementResult]:
    """Select a portfolio from raw proposals.

    Take one targeted proposal per direction first, preserving direction order,
    then fill remaining slots with bold proposals in original order.

    When *incumbent_template* is provided, the portfolio is post-checked for structural
    novelty: if every selected proposal shares the incumbent's structural signature,
    swap the lowest-priority non-targeted pick for a structurally-novel candidate when one exists.
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
            break

    if len(selected) < num_experiments:
        bold_candidates = [r for r in results if r.risk_level == "bold" and id(r) not in seen_ids]
        for candidate in bold_candidates:
            selected.append(candidate)
            seen_ids.add(id(candidate))
            if len(selected) >= num_experiments:
                break

    if len(selected) < num_experiments:
        for candidate in results:
            if id(candidate) in seen_ids:
                continue
            selected.append(candidate)
            seen_ids.add(id(candidate))
            if len(selected) >= num_experiments:
                break

    selected = selected[:num_experiments]

    if incumbent_template is not None and selected:
        has_novel = any(_is_structurally_novel(r.template, incumbent_template) for r in selected)
        if not has_novel:
            novel_candidate = next(
                (
                    r
                    for r in results
                    if id(r) not in {id(s) for s in selected} and _is_structurally_novel(r.template, incumbent_template)
                ),
                None,
            )
            if novel_candidate is not None:
                swap_positions = [i for i, s in enumerate(selected) if s.risk_level == "bold"]
                if not swap_positions:
                    swap_positions = [len(selected) - 1]
                swap_idx = swap_positions[-1]
                replaced = selected[swap_idx]
                logger.info(
                    "Structural novelty swap: replacing '%s' (shares incumbent signature) with '%s' (novel)",
                    replaced.hypothesis[:80],
                    novel_candidate.hypothesis[:80],
                )
                selected[swap_idx] = novel_candidate
            else:
                logger.warning(
                    "Portfolio has no structurally novel proposals vs incumbent and no swap candidate available — "
                    "mode-collapse risk for this iteration"
                )

    return selected


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
    is_first_iteration: bool = False,
    iteration: int = 0,
    plateau_counter: int = 0,
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
        vision_feedback_word_limit=300,
        roundtrip_feedback_word_limit=400,
        include_roundtrip_feedback=True,
        iteration=iteration,
        plateau_counter=plateau_counter,
    )
    user = _brainstorm_user(
        shared_user, num_sketches=num_sketches, has_feedback=bool(vision_feedback or roundtrip_feedback)
    )
    logger.info(
        "Brainstorming %d experiment sketches (%s) — context: ~%d words", num_sketches, model, len(user.split())
    )
    sketches, converged = await client.call_json(
        model=model,
        system=_brainstorm_system(current_template, num_sketches=num_sketches, is_first_iteration=is_first_iteration),
        user=user,
        validator=lambda data: validate_brainstorm_payload(data, num_sketches=num_sketches),
        response_name="brainstorm",
        schema_hint=schema_hint("brainstorm"),
        response_schema=response_schema("brainstorm"),
        max_tokens=40000,
        repair_retries=2,
        temperature=0.9,
        reasoning_effort="high",
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
            response_schema=response_schema("ranking"),
            max_tokens=10000,
            repair_retries=1,
            final_failure_log_level=logging.INFO,
            temperature=0.1,
            reasoning_effort="low",
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
    is_first_iteration: bool = False,
    iteration: int = 0,
    plateau_counter: int = 0,
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
        vision_feedback_word_limit=300,
        roundtrip_feedback_word_limit=400,
        include_roundtrip_feedback=not bool(vision_feedback),
        iteration=iteration,
        plateau_counter=plateau_counter,
    )
    system = _expand_system(current_template, is_first_iteration=is_first_iteration)
    tasks = [
        client.call_json(
            model=model,
            system=system,
            user=_expand_user(shared_user, sketch),
            validator=validate_expansion_payload,
            response_name=f"expansion_{idx}",
            schema_hint=schema_hint("expansion"),
            response_schema=response_schema("expansion"),
            max_tokens=24000,
            repair_retries=2,
            temperature=0.3,
            reasoning_effort="high",
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
        # Programmatic post-expand verification
        issues = _validate_expanded_template(raw, current_template)
        if issues:
            logger.warning(
                "Post-expand validation issues for '%s': %s",
                raw.hypothesis[:60],
                "; ".join(issues),
            )
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
    is_first_iteration: bool = False,
    iteration: int = 0,
    plateau_counter: int = 0,
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
        is_first_iteration=is_first_iteration,
        iteration=iteration,
        plateau_counter=plateau_counter,
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
        is_first_iteration=is_first_iteration,
        iteration=iteration,
        plateau_counter=plateau_counter,
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

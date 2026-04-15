"""Experiment proposal helpers for one iteration."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from art_style_search.contracts import ExperimentProposal, ExperimentSketch
from art_style_search.prompt._parse import validate_template
from art_style_search.prompt.experiments import (
    brainstorm_experiment_sketches,
    enforce_hypothesis_diversity,
    expand_experiment_sketches,
    rank_experiment_sketches,
    select_experiment_portfolio,
)
from art_style_search.scoring import classify_hypothesis
from art_style_search.types import ConvergenceReason, LoopState, get_category_names
from art_style_search.workflow.context import RunContext
from art_style_search.workflow.policy import _should_honor_stop

logger = logging.getLogger(__name__)

_STRUCTURAL_CHANGE_FIELDS = ("caption_sections", "caption_length_target", "negative_prompt")
_CATEGORY_ALIAS_TO_FIELDS: dict[str, tuple[str, ...]] = {
    "caption_structure": ("caption_sections", "caption_length_target"),
}


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _infer_changed_targets(current_template, proposed_template) -> list[str]:
    inferred: list[str] = []

    if current_template.caption_sections != proposed_template.caption_sections:
        inferred.append("caption_sections")
    if current_template.caption_length_target != proposed_template.caption_length_target:
        inferred.append("caption_length_target")
    if (current_template.negative_prompt or "") != (proposed_template.negative_prompt or ""):
        inferred.append("negative_prompt")

    current_by_name = {section.name: section for section in current_template.sections}
    proposed_by_name = {section.name: section for section in proposed_template.sections}

    removed_names = [section.name for section in current_template.sections if section.name not in proposed_by_name]
    added_names = [section.name for section in proposed_template.sections if section.name not in current_by_name]
    inferred.extend(removed_names)
    if not removed_names:
        inferred.extend(added_names)

    common_changed_names = [
        section.name
        for section in current_template.sections
        if section.name in proposed_by_name
        and (
            section.description != proposed_by_name[section.name].description
            or section.value != proposed_by_name[section.name].value
        )
    ]
    if not removed_names and not added_names and len(common_changed_names) == 1:
        inferred.append(common_changed_names[0])

    return _ordered_unique(inferred)


def _recover_proposal_change_metadata(
    proposal: ExperimentProposal,
    current_template,
) -> tuple[str, list[str], bool]:
    original_targets = _ordered_unique(
        ([proposal.changed_section] if proposal.changed_section else []) + list(proposal.changed_sections or [])
    )
    inferred_targets = _infer_changed_targets(current_template, proposal.template)
    inferred_structural_targets = [name for name in inferred_targets if name in _STRUCTURAL_CHANGE_FIELDS]

    allowed_names = (
        {section.name for section in current_template.sections}
        | {section.name for section in proposal.template.sections}
        | set(_STRUCTURAL_CHANGE_FIELDS)
    )

    recovered_targets: list[str] = []
    used_alias = False
    for target in original_targets:
        if target in allowed_names:
            recovered_targets.append(target)
            continue
        alias_targets = _CATEGORY_ALIAS_TO_FIELDS.get(target, ())
        if alias_targets:
            used_alias = True
            recovered_targets.extend(
                name for name in alias_targets if name in inferred_structural_targets and name not in recovered_targets
            )

    category_alias_targets = _CATEGORY_ALIAS_TO_FIELDS.get(proposal.target_category, ())
    if category_alias_targets:
        used_alias = True
        recovered_targets.extend(
            name
            for name in category_alias_targets
            if name in inferred_structural_targets and name not in recovered_targets
        )

    if not recovered_targets:
        recovered_targets = list(inferred_targets)
    elif used_alias:
        recovered_targets = _ordered_unique(recovered_targets + inferred_structural_targets)
    else:
        recovered_targets = _ordered_unique(recovered_targets)

    recovered_changed_section = recovered_targets[0] if recovered_targets else ""
    was_recovered = (
        recovered_changed_section != (original_targets[0] if original_targets else "")
        or recovered_targets != original_targets
    )
    return recovered_changed_section, recovered_targets, was_recovered


def _dedupe_ranked_sketches(
    sketches: list[ExperimentSketch],
    current_template,
) -> list[ExperimentSketch]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[ExperimentSketch] = []
    category_names = get_category_names(current_template)

    for sketch in sketches:
        category = sketch.target_category or classify_hypothesis(sketch.hypothesis, category_names)
        key = (
            category,
            sketch.failure_mechanism.strip().lower(),
            sketch.intervention_type.strip().lower(),
        )
        if key in seen:
            logger.warning(
                "Dropping duplicate ranked sketch (category=%s, mechanism=%s, intervention=%s): %s",
                key[0],
                key[1] or "<none>",
                key[2] or "<none>",
                sketch.hypothesis[:80],
            )
            continue
        seen.add(key)
        deduped.append(sketch)

    return deduped


async def _propose_iteration_experiments(
    state: LoopState,
    ctx: RunContext,
    vision_fb: str,
    roundtrip_fb: str,
    caption_diffs: str,
) -> tuple[list[ExperimentProposal], bool]:
    """Phase 1: brainstorm sketches, rank them, expand survivors, then select a portfolio."""
    requested_sketches = max(ctx.config.raw_proposals * 2, ctx.config.raw_proposals)
    sketches, converged = await brainstorm_experiment_sketches(
        state.style_profile,
        state.current_template,
        state.knowledge_base,
        state.best_metrics,
        state.last_iteration_results,
        client=ctx.reasoning_client,
        model=ctx.config.reasoning_model,
        num_sketches=requested_sketches,
        vision_feedback=vision_fb,
        roundtrip_feedback=roundtrip_fb,
        caption_diffs=caption_diffs,
        is_first_iteration=state.iteration == 0,
        iteration=state.iteration,
        plateau_counter=state.plateau_counter,
    )
    logger.info("Brainstorm step returned %d sketches (requested=%d)", len(sketches), requested_sketches)

    if converged:
        if _should_honor_stop(state, ctx, reason="reasoning model emitted stop"):
            logger.info("Reasoning model signaled convergence during brainstorm — honored")
            state.converged = True
            state.convergence_reason = ConvergenceReason.REASONING_STOP
            return [], True
        logger.info("Reasoning model signaled convergence during brainstorm — guard rejected")

    if not sketches:
        if _should_honor_stop(state, ctx, reason="no experiment sketches proposed"):
            logger.warning("No experiment sketches proposed — honoring stop")
            state.converged = True
            state.convergence_reason = ConvergenceReason.REASONING_STOP
            return [], True
        logger.warning("No experiment sketches proposed — guard rejected, continuing with empty batch")
        return [], False

    ranked_sketches = await rank_experiment_sketches(
        sketches,
        state.knowledge_base,
        state.best_metrics,
        client=ctx.reasoning_client,
        model=ctx.config.reasoning_model,
    )
    ranked_sketches = ranked_sketches[: ctx.config.raw_proposals]
    logger.info("Ranking step kept %d/%d sketches for expansion", len(ranked_sketches), len(sketches))

    ranked_sketches = _dedupe_ranked_sketches(ranked_sketches, state.current_template)
    logger.info("Sketch diversity filter kept %d sketches for expansion", len(ranked_sketches))

    refinements = await expand_experiment_sketches(
        state.style_profile,
        state.current_template,
        state.knowledge_base,
        state.best_metrics,
        state.last_iteration_results,
        client=ctx.reasoning_client,
        model=ctx.config.reasoning_model,
        sketches=ranked_sketches,
        vision_feedback=vision_fb,
        roundtrip_feedback=roundtrip_fb,
        caption_diffs=caption_diffs,
        is_first_iteration=state.iteration == 0,
        iteration=state.iteration,
        plateau_counter=state.plateau_counter,
    )
    logger.info("Expand step returned %d refinement proposals", len(refinements))

    refinements = enforce_hypothesis_diversity(refinements, state.current_template)
    refinements = select_experiment_portfolio(refinements, num_experiments=ctx.config.num_branches, num_directions=3)
    proposals: list[ExperimentProposal] = []
    for refinement in refinements:
        if refinement.should_stop:
            if _should_honor_stop(state, ctx, reason="reasoning model emitted stop"):
                logger.info("Reasoning model signaled convergence — honored")
                state.converged = True
                state.convergence_reason = ConvergenceReason.REASONING_STOP
                return [], True
            refinement.should_stop = False
        proposals.append(
            ExperimentProposal(
                template=refinement.template,
                hypothesis=refinement.hypothesis,
                experiment_desc=refinement.experiment,
                builds_on=refinement.builds_on,
                open_problems=refinement.open_problems,
                lessons=refinement.lessons,
                analysis=refinement.analysis,
                template_changes=refinement.template_changes,
                changed_section=refinement.changed_section,
                changed_sections=list(refinement.changed_sections or []),
                target_category=refinement.target_category,
                direction_id=refinement.direction_id,
                direction_summary=refinement.direction_summary,
                failure_mechanism=refinement.failure_mechanism,
                intervention_type=refinement.intervention_type,
                risk_level=refinement.risk_level,
                expected_primary_metric=refinement.expected_primary_metric,
                expected_tradeoff=refinement.expected_tradeoff,
            )
        )

    if not proposals:
        if _should_honor_stop(state, ctx, reason="no experiments proposed"):
            logger.warning("No experiments proposed — honoring stop")
            state.converged = True
            state.convergence_reason = ConvergenceReason.REASONING_STOP
            return [], True
        logger.warning("No experiments proposed — guard rejected, continuing with empty batch")
        return [], False

    valid_proposals: list[ExperimentProposal] = []
    direction_counts: defaultdict[str, int] = defaultdict(int)
    rejection_counts: Counter[str] = Counter()
    recovered_count = 0
    for proposal in proposals:
        changed_section, changed_sections, was_recovered = _recover_proposal_change_metadata(
            proposal, state.current_template
        )
        proposal.changed_section = changed_section
        proposal.changed_sections = changed_sections
        if was_recovered:
            recovered_count += 1
        errors = validate_template(
            proposal.template,
            proposal.changed_section,
            proposal.changed_sections,
            proposal.risk_level,
            state.current_template,
        )
        if proposal.risk_level == "targeted" and len(proposal.changed_sections or []) != 1:
            errors = [*errors, "targeted experiments must change exactly 1 section"]
        if errors:
            logger.warning("Skipping invalid proposal (hyp: %.80s): %s", proposal.hypothesis, "; ".join(errors))
            for error in errors:
                rejection_counts[error.split(":")[0]] += 1
            continue
        direction_counts[proposal.direction_id or ""] += 1
        valid_proposals.append(proposal)
    logger.info(
        "Proposal validation summary: raw=%d recovered=%d kept=%d rejected=%d",
        len(proposals),
        recovered_count,
        len(valid_proposals),
        len(proposals) - len(valid_proposals),
    )
    if rejection_counts:
        top_reasons = ", ".join(f"{reason} x{count}" for reason, count in rejection_counts.most_common(3))
        logger.warning("Top proposal rejection reasons: %s", top_reasons)
    if valid_proposals:
        logger.info("Selected proposal portfolio across %d directions", len([k for k in direction_counts if k]))
    return valid_proposals, False

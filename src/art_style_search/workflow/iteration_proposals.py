"""Experiment proposal helpers for one iteration."""

from __future__ import annotations

import logging
from collections import defaultdict

from art_style_search.contracts import ExperimentProposal
from art_style_search.prompt._parse import validate_template
from art_style_search.prompt.experiments import (
    enforce_hypothesis_diversity,
    propose_experiments,
    select_experiment_portfolio,
)
from art_style_search.types import ConvergenceReason, LoopState
from art_style_search.workflow.context import RunContext
from art_style_search.workflow.policy import _should_honor_stop

logger = logging.getLogger(__name__)


async def _propose_iteration_experiments(
    state: LoopState,
    ctx: RunContext,
    vision_fb: str,
    roundtrip_fb: str,
    caption_diffs: str,
) -> tuple[list[ExperimentProposal], bool]:
    """Phase 1: propose a raw batch, then select a 3-direction portfolio."""
    refinements = await propose_experiments(
        state.style_profile,
        state.current_template,
        state.knowledge_base,
        state.best_metrics,
        state.last_iteration_results,
        client=ctx.reasoning_client,
        model=ctx.config.reasoning_model,
        num_experiments=ctx.config.raw_proposals,
        vision_feedback=vision_fb,
        roundtrip_feedback=roundtrip_fb,
        caption_diffs=caption_diffs,
    )

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
    for proposal in proposals:
        errors = validate_template(
            proposal.template,
            proposal.changed_section,
            proposal.changed_sections,
            proposal.risk_level,
        )
        if errors:
            logger.warning("Skipping invalid proposal (hyp: %.80s): %s", proposal.hypothesis, "; ".join(errors))
            continue
        direction_counts[proposal.direction_id or ""] += 1
        valid_proposals.append(proposal)
    if valid_proposals:
        logger.info("Selected proposal portfolio across %d directions", len([k for k in direction_counts if k]))
    return valid_proposals, False

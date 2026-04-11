"""Experiment proposal helpers for one iteration."""

from __future__ import annotations

import logging

from art_style_search.contracts import ExperimentProposal
from art_style_search.prompt._parse import validate_template
from art_style_search.prompt.experiments import enforce_hypothesis_diversity, propose_experiments
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
    """Phase 1: propose N experiments, dedup by category, convert to ExperimentProposal."""
    refinements = await propose_experiments(
        state.style_profile,
        state.current_template,
        state.knowledge_base,
        state.best_metrics,
        state.last_iteration_results,
        client=ctx.reasoning_client,
        model=ctx.config.reasoning_model,
        num_experiments=ctx.config.num_branches,
        vision_feedback=vision_fb,
        roundtrip_feedback=roundtrip_fb,
        caption_diffs=caption_diffs,
    )

    refinements = enforce_hypothesis_diversity(refinements, state.current_template)
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
                target_category=refinement.target_category,
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
    for proposal in proposals:
        errors = validate_template(proposal.template, proposal.changed_section)
        if errors:
            logger.warning("Skipping invalid proposal (hyp: %.80s): %s", proposal.hypothesis, "; ".join(errors))
            continue
        valid_proposals.append(proposal)
    return valid_proposals, False

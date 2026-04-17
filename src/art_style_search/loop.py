"""Main orchestration loop façade.

The concrete implementations now live under ``art_style_search.workflow``.
This module keeps the public ``run()`` entrypoint plus compatibility exports
for tests and local tooling that still reach into selected helpers.
"""

from __future__ import annotations

import asyncio
import logging

from art_style_search.config import Config
from art_style_search.experiment import run_experiment
from art_style_search.prompt import (
    enforce_hypothesis_diversity,
    propose_experiments,
    propose_initial_templates,
    review_iteration,
    synthesize_templates,
    validate_template,
)
from art_style_search.state import load_state
from art_style_search.types import ConvergenceReason, LoopState
from art_style_search.workflow.context import (
    RunContext,
    _discover_images,
    _finalize_run,
    _setup_run_context,
    ensure_manifest,
)
from art_style_search.workflow.iteration_context import _build_iteration_context
from art_style_search.workflow.iteration_execution import (
    IterationRanking,
    _confirmatory_validation,
    _run_experiments_parallel,
    _run_independent_review,
    _run_pairwise_comparison,
    _run_synthesis_experiment,
    _score_and_rank,
    _synthesize_reasoning,
)
from art_style_search.workflow.iteration_persistence import (
    _record_iteration_state,
    _update_knowledge_base_for_iteration,
)
from art_style_search.workflow.iteration_proposals import _propose_iteration_experiments
from art_style_search.workflow.policy import (
    _apply_iteration_result,
    _check_plateau_convergence,
)
from art_style_search.workflow.zero_step import _zero_step, maybe_rebuild_canon_on_resume

logger = logging.getLogger(__name__)

__all__ = [
    "IterationRanking",
    "RunContext",
    "enforce_hypothesis_diversity",
    "propose_experiments",
    "propose_initial_templates",
    "review_iteration",
    "run",
    "run_experiment",
    "synthesize_templates",
    "validate_template",
]


async def run(config: Config) -> LoopState:
    """Run the full optimization loop."""
    ctx = await _setup_run_context(config)

    all_ref_paths = _discover_images(config.reference_dir)
    if not all_ref_paths:
        msg = f"No images found in {config.reference_dir}"
        raise FileNotFoundError(msg)
    logger.info("Found %d reference images", len(all_ref_paths))

    ensure_manifest(config)

    state = load_state(config.state_file)
    if state is not None:
        logger.info("Resumed from iteration %d with %d fixed references", state.iteration, len(state.fixed_references))
        if state.converged:
            logger.info("Previous run already converged (%s) — skipping loop", state.convergence_reason)
            return _finalize_run(state, ctx)
        if await maybe_rebuild_canon_on_resume(state, ctx):
            from art_style_search.state import save_state

            save_state(state, config.state_file)
    else:
        state = await _zero_step(ctx, all_ref_paths)

    for iteration in range(state.iteration, config.max_iterations):
        state.iteration = iteration
        logger.info("=== Iteration %d/%d ===", iteration + 1, config.max_iterations)

        vision_fb, roundtrip_fb, caption_diffs = _build_iteration_context(state)
        proposals, converged = await _propose_iteration_experiments(state, ctx, vision_fb, roundtrip_fb, caption_diffs)
        if converged:
            break

        exp_results = await _run_experiments_parallel(state, ctx, proposals, iteration)
        if not exp_results:
            logger.warning("All experiments failed this iteration")
            state.plateau_counter += 1
            state.iteration = iteration + 1
            from art_style_search.state import save_state

            save_state(state, config.state_file)
            if state.plateau_counter >= config.plateau_window:
                state.converged = True
                state.convergence_reason = ConvergenceReason.PLATEAU
                break
            continue

        ranking = _score_and_rank(exp_results, state)
        synth_result, pairwise_result, review_result = await asyncio.gather(
            _synthesize_reasoning(ranking, state, ctx),
            _run_pairwise_comparison(ranking, state, ctx),
            _run_independent_review(ranking, proposals, state, ctx),
            return_exceptions=True,
        )
        if isinstance(pairwise_result, BaseException):
            logger.warning("Pairwise comparison failed — skipping: %s", pairwise_result)
        if isinstance(review_result, BaseException):
            logger.warning("Independent review failed — skipping: %s", review_result)

        if isinstance(synth_result, BaseException):
            logger.warning("Synthesis reasoning failed — skipping: %s", synth_result)
        elif synth_result is not None:
            try:
                await _run_synthesis_experiment(synth_result, ranking, state, ctx, iteration)
            except Exception:
                logger.warning("Synthesis experiment failed — continuing with individual experiments", exc_info=True)

        try:
            await _confirmatory_validation(ranking, state, ctx, iteration)
        except Exception:
            logger.warning("Confirmatory validation failed — falling back to single-pass", exc_info=True)

        baseline_metrics = state.best_metrics
        decision = _apply_iteration_result(state, ranking, ctx.config)
        _update_knowledge_base_for_iteration(state, ranking, proposals, baseline_metrics, iteration, decision)
        _record_iteration_state(state, ranking, iteration, ctx)

        if _check_plateau_convergence(state, ctx):
            break
    else:
        state.converged = True
        state.convergence_reason = ConvergenceReason.MAX_ITERATIONS

    return _finalize_run(state, ctx)

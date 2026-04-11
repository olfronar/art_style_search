"""Promotion and convergence policy helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from art_style_search.state import append_promotion_log
from art_style_search.types import ConvergenceReason, IterationResult, LoopState, PromotionDecision, PromotionTestResult
from art_style_search.utils import CATEGORY_SYNONYMS

if TYPE_CHECKING:
    from art_style_search.config import Config
    from art_style_search.workflow.context import RunContext
    from art_style_search.workflow.iteration import IterationRanking

from art_style_search.scoring import composite_score

logger = logging.getLogger(__name__)

_EXPLORATION_MIN_PLATEAU = 2
_EXPLORATION_CADENCE = 2
_EXPLORATION_RESET_PLATEAU = 1
_MIN_ITER_FRACTION_FOR_STOP = 0.5


def _apply_best_result(state: LoopState, result: IterationResult) -> None:
    """Update state with a genuine improvement."""
    result.kept = True
    state.current_template = result.template
    state.best_template = result.template
    state.best_metrics = result.aggregated
    global_score = composite_score(state.global_best_metrics) if state.global_best_metrics else float("-inf")
    if composite_score(result.aggregated) > global_score:
        state.global_best_prompt = result.rendered_prompt
        state.global_best_metrics = result.aggregated


def _apply_exploration_result(state: LoopState, result: IterationResult) -> None:
    """Adopt a result for exploration while preserving the best known metrics."""
    result.kept = True
    state.current_template = result.template
    state.best_template = result.template


def _candidate_results_for_validation(ranking: IterationRanking) -> list[IterationResult]:
    """Return the top proposal candidates plus synthesis candidate, if present."""
    proposal_results = [
        result for result in ranking.exp_results if ranking.synth_result is None or result is not ranking.synth_result
    ]
    sorted_proposals = sorted(proposal_results, key=lambda result: ranking.adaptive_scores[id(result)], reverse=True)
    candidates = sorted_proposals[:2]
    if ranking.synth_result is not None and ranking.synth_result not in candidates:
        candidates.append(ranking.synth_result)
    return candidates


def _log_promotion_decision(
    state: LoopState,
    ranking: IterationRanking,
    decision: str,
    reason: str,
    config: Config,
    *,
    candidate: IterationResult | None = None,
    candidate_score: float | None = None,
    replicate_scores: list[float] | None = None,
    promotion_test: PromotionTestResult | None = None,
) -> None:
    """Log a promotion decision to promotion_log.jsonl."""
    selected = candidate or ranking.best_exp
    score = candidate_score if candidate_score is not None else composite_score(selected.aggregated)
    test_result = (
        promotion_test
        if promotion_test is not None
        else (ranking.promotion_test if selected is ranking.best_exp else None)
    )
    decision_record = PromotionDecision(
        iteration=state.iteration,
        candidate_score=score,
        baseline_score=ranking.baseline_score,
        epsilon=ranking.epsilon,
        delta=score - ranking.baseline_score,
        decision=decision,
        reason=reason,
        candidate_branch_id=selected.branch_id,
        candidate_hypothesis=selected.hypothesis,
        replicate_scores=replicate_scores,
        p_value=test_result.p_value if test_result is not None else None,
        test_statistic=test_result.statistic if test_result is not None else None,
    )
    append_promotion_log(decision_record, config.run_dir / "promotion_log.jsonl")


def _apply_iteration_result(state: LoopState, ranking: IterationRanking, config: Config) -> str:
    """Decide improvement vs plateau and update state accordingly."""
    improved = ranking.best_score > ranking.baseline_score + ranking.epsilon

    if improved and config.protocol == "rigorous":
        promotion_test = ranking.promotion_test
        if promotion_test is not None and not promotion_test.passed:
            logger.info(
                "Epsilon check passed but statistical test failed (p=%.4f) — rejecting promotion",
                promotion_test.p_value,
            )
            improved = False

    if improved:
        _apply_best_result(state, ranking.best_exp)
        state.plateau_counter = 0
        _log_promotion_decision(
            state,
            ranking,
            "promoted",
            "Exceeded baseline + epsilon",
            config,
            replicate_scores=ranking.best_replicate_scores,
        )
        return "promoted"

    state.plateau_counter += 1
    can_explore = (
        state.plateau_counter >= _EXPLORATION_MIN_PLATEAU
        and state.plateau_counter % _EXPLORATION_CADENCE == 0
        and len(ranking.exp_results) >= 2
    )
    if can_explore:
        ranked = sorted(ranking.exp_results, key=lambda result: ranking.adaptive_scores[id(result)], reverse=True)
        second_best = ranked[1]
        logger.info("Exploration: adopting second-best experiment to escape potential local optimum")
        _apply_exploration_result(state, second_best)
        state.plateau_counter = _EXPLORATION_RESET_PLATEAU
        _log_promotion_decision(
            state,
            ranking,
            "exploration",
            "Plateau escape via second-best",
            config,
            candidate=second_best,
        )
        return "exploration"

    _log_promotion_decision(
        state,
        ranking,
        "rejected",
        f"Delta {ranking.best_score - ranking.baseline_score:.5f} < epsilon {ranking.epsilon:.5f}",
        config,
        replicate_scores=ranking.best_replicate_scores,
    )
    return "rejected"


def _check_plateau_convergence(state: LoopState, ctx: RunContext) -> bool:
    """Return True if the plateau window has been hit."""
    if state.plateau_counter >= ctx.config.plateau_window:
        logger.info("Plateau detected (%d iterations without improvement)", state.plateau_counter)
        state.converged = True
        state.convergence_reason = ConvergenceReason.PLATEAU
        return True
    return False


def _should_honor_stop(state: LoopState, ctx: RunContext, reason: str) -> bool:
    """Gate the reasoning model's stop signal behind substantive conditions."""
    min_iter_floor = ctx.config.max_iterations * _MIN_ITER_FRACTION_FOR_STOP
    if state.iteration + 1 < min_iter_floor:
        logger.info(
            "Rejecting stop (%s): iteration %d below floor %.1f",
            reason,
            state.iteration + 1,
            min_iter_floor,
        )
        return False

    plateau_floor = max(ctx.config.plateau_window - 1, 2)
    if state.plateau_counter < plateau_floor:
        logger.info(
            "Rejecting stop (%s): plateau_counter %d below floor %d",
            reason,
            state.plateau_counter,
            plateau_floor,
        )
        return False

    tried = {cat for cat, progress in state.knowledge_base.categories.items() if progress.hypothesis_ids}
    untried = sorted(cat for cat in CATEGORY_SYNONYMS if cat not in tried)
    if untried:
        logger.info("Rejecting stop (%s): unexplored categories=%s", reason, untried)
        return False

    return True

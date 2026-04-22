"""Promotion and convergence policy helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from art_style_search.knowledge import IterationDecision
from art_style_search.state import append_promotion_log
from art_style_search.types import (
    AggregatedMetrics,
    ConvergenceReason,
    IterationResult,
    LoopState,
    PromotionDecision,
    Protocol,
    ScoringFunction,
)
from art_style_search.utils import CATEGORY_SYNONYMS

if TYPE_CHECKING:
    from art_style_search.config import Config
    from art_style_search.workflow.context import RunContext
    from art_style_search.workflow.iteration_execution import IterationRanking

from art_style_search.scoring import composite_score, headroom_composite_score, replicate_promotion_decision


def _promotion_score(m: AggregatedMetrics, *, protocol: Protocol | str) -> float:
    """A6 dispatch: classic = headroom-weighted (saturated axes carry no marginal utility),
    short = plain composite (foundation pass hasn't saturated anything yet).

    A1 composes with A6 under classic — every scalar in ``replicate_promotion_decision``
    must pass through here so dominance + median-epsilon never mixes scales.
    """
    if protocol == "classic":
        return headroom_composite_score(m)
    return composite_score(m)


def _scoring_function_name(protocol: Protocol | str) -> ScoringFunction:
    """Audit label for ``promotion_log.jsonl`` — mirrors ``_promotion_score``'s branch."""
    return "headroom" if protocol == "classic" else "composite"


logger = logging.getLogger(__name__)

_EXPLORATION_MIN_PLATEAU = 2
_EXPLORATION_CADENCE = 2
_EXPLORATION_RESET_PLATEAU = 1
_MIN_ITER_FRACTION_FOR_STOP = 0.5


def _apply_best_result(state: LoopState, result: IterationResult) -> None:
    """Update state with a genuine improvement.

    ``global_best_*`` guard uses the run's current scoring function so a classic iteration
    never regresses to a short-protocol incumbent on a mixed scale.
    """
    result.kept = True
    state.current_template = result.template
    state.best_template = result.template
    state.best_metrics = result.aggregated
    score = _promotion_score(result.aggregated, protocol=state.protocol)
    global_score = (
        _promotion_score(state.global_best_metrics, protocol=state.protocol)
        if state.global_best_metrics
        else float("-inf")
    )
    if score > global_score:
        state.global_best_prompt = result.rendered_prompt
        state.global_best_metrics = result.aggregated


def _apply_exploration_result(state: LoopState, result: IterationResult) -> None:
    """Adopt a result for exploration while preserving the best known metrics.

    Only ``current_template`` is updated so that proposals diverge from the
    exploration direction.  ``best_template`` stays in sync with
    ``best_metrics``.
    """
    result.kept = True
    state.current_template = result.template


def _select_exploration_candidate(state: LoopState, ranking: IterationRanking) -> tuple[IterationResult, str]:
    """Prefer bold or untried-mechanism candidates over raw second-best."""
    ranked = sorted(ranking.exp_results, key=lambda result: ranking.adaptive_scores[id(result)], reverse=True)
    fallback = ranked[1]
    historical_mechanisms = {hyp.failure_mechanism for hyp in state.knowledge_base.hypotheses if hyp.failure_mechanism}
    exploration_pool = ranked[1:]

    bold_untried = [
        result
        for result in exploration_pool
        if result.risk_level == "bold"
        and result.failure_mechanism
        and result.failure_mechanism not in historical_mechanisms
    ]
    if bold_untried:
        return bold_untried[0], "Plateau escape via bold untried mechanism"

    bold_candidates = [result for result in exploration_pool if result.risk_level == "bold"]
    if bold_candidates:
        return bold_candidates[0], "Plateau escape via bold candidate"

    untried = [
        result
        for result in exploration_pool
        if result.failure_mechanism and result.failure_mechanism not in historical_mechanisms
    ]
    if untried:
        return untried[0], "Plateau escape via untried mechanism"

    return fallback, "Plateau escape via second-best"


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
) -> None:
    """Log a promotion decision to promotion_log.jsonl."""
    selected = candidate or ranking.best_exp
    score = (
        candidate_score
        if candidate_score is not None
        else _promotion_score(selected.aggregated, protocol=config.protocol)
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
        scoring_function=_scoring_function_name(config.protocol),
    )
    append_promotion_log(decision_record, config.run_dir / "promotion_log.jsonl")


def _apply_iteration_result(state: LoopState, ranking: IterationRanking, config: Config) -> IterationDecision:
    """Decide improvement vs plateau and update state accordingly.

    When replicate scores are present on the ranking (A1 paired-replicate gate), use the
    dominance + effect-size decision; otherwise fall back to the single-shot epsilon check.
    """
    if ranking.best_replicate_scores is not None and ranking.baseline_replicate_scores is not None:
        improved = (
            replicate_promotion_decision(
                ranking.best_replicate_scores,
                ranking.baseline_replicate_scores,
                epsilon=ranking.epsilon,
            )
            == "promoted"
        )
    else:
        improved = ranking.best_score > ranking.baseline_score + ranking.epsilon

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
        candidate, reason = _select_exploration_candidate(state, ranking)
        logger.info("Exploration: adopting branch %d (%s)", candidate.branch_id, reason)
        _apply_exploration_result(state, candidate)
        state.plateau_counter = _EXPLORATION_RESET_PLATEAU
        _log_promotion_decision(
            state,
            ranking,
            "exploration",
            reason,
            config,
            candidate=candidate,
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

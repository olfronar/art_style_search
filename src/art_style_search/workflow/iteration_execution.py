"""Execution and analysis helpers for one iteration."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import partial

from art_style_search.contracts import ExperimentProposal
from art_style_search.experiment import (
    collect_experiment_results,
    replicate_experiment,
    run_experiment,
)
from art_style_search.prompt._parse import validate_template
from art_style_search.prompt.review import review_iteration
from art_style_search.prompt.synthesis import synthesize_templates
from art_style_search.scoring import (
    DELTA_METRIC_LABELS,
    IMPROVEMENT_EPSILON,
    adaptive_composite_score,
    composite_score,
    improvement_epsilon,
    metric_deltas,
)
from art_style_search.types import IterationResult, LoopState, PromptTemplate
from art_style_search.utils import build_ref_gen_pairs
from art_style_search.workflow.context import RunContext
from art_style_search.workflow.policy import _promotion_score

logger = logging.getLogger(__name__)

# A1: incumbent replicates use a dedicated branch_id so they don't collide with proposal indices.
_INCUMBENT_BRANCH_ID = 900


@dataclass
class IterationRanking:
    """Ranking state that crosses phase boundaries within one iteration."""

    exp_results: list[IterationResult]
    adaptive_scores: dict[int, float]
    best_exp: IterationResult
    best_score: float
    baseline_score: float
    epsilon: float
    synth_result: IterationResult | None = None
    # A1 paired-replicate gate: when ``replicates > 1``, ``_run_replicate_gate`` populates both
    # lists with per-replicate composite scores. ``_apply_iteration_result`` in policy.py
    # switches to ``replicate_promotion_decision`` when both are non-None.
    best_replicate_scores: list[float] | None = None
    baseline_replicate_scores: list[float] | None = None


async def _run_experiments_parallel(
    state: LoopState,
    ctx: RunContext,
    proposals: list[ExperimentProposal],
    iteration: int,
) -> list[IterationResult]:
    """Phase 2: gather all experiment runs in parallel."""
    exp_tasks = [
        run_experiment(
            experiment_id=i,
            template=proposal.template,
            iteration=iteration,
            fixed_refs=state.fixed_references,
            config=ctx.config,
            last_results=state.last_iteration_results,
            hypothesis=proposal.hypothesis,
            experiment_desc=proposal.experiment_desc,
            analysis=proposal.analysis,
            template_changes=proposal.template_changes,
            changed_section=proposal.changed_section,
            changed_sections=proposal.changed_sections,
            target_category=proposal.target_category,
            direction_id=proposal.direction_id,
            direction_summary=proposal.direction_summary,
            failure_mechanism=proposal.failure_mechanism,
            intervention_type=proposal.intervention_type,
            risk_level=proposal.risk_level,
            expected_primary_metric=proposal.expected_primary_metric,
            expected_tradeoff=proposal.expected_tradeoff,
            canon_ops=proposal.canon_ops,
            services=ctx.services,
        )
        for i, proposal in enumerate(proposals)
    ]
    return collect_experiment_results(await asyncio.gather(*exp_tasks, return_exceptions=True), "Experiment")


def _score_and_rank(exp_results: list[IterationResult], state: LoopState) -> IterationRanking:
    """Phase 3: compute aggregates, adaptive scores, pick best.

    ``adaptive_composite_score`` is relative (min-max across the peer set) and stays unchanged —
    its job is ranking experiments against each other, not gating promotion. ``best_score`` /
    ``baseline_score`` are the absolute scores the promotion gate compares with epsilon, so they
    use ``_promotion_score`` — headroom-weighted under classic, plain composite under short.
    """
    all_agg = [result.aggregated for result in exp_results]
    adaptive_scores = {id(result): adaptive_composite_score(result.aggregated, all_agg) for result in exp_results}
    best_exp = max(exp_results, key=lambda result: adaptive_scores[id(result)])
    best_score = _promotion_score(best_exp.aggregated, protocol=state.protocol)
    baseline_score = (
        _promotion_score(state.best_metrics, protocol=state.protocol) if state.best_metrics else float("-inf")
    )
    return IterationRanking(
        exp_results=exp_results,
        adaptive_scores=adaptive_scores,
        best_exp=best_exp,
        best_score=best_score,
        baseline_score=baseline_score,
        epsilon=improvement_epsilon(baseline_score),
    )


async def _run_pairwise_comparison(ranking: IterationRanking, state: LoopState, ctx: RunContext) -> None:
    """Phase 3.7: pairwise comparison of top experiments."""
    exp_results = ranking.exp_results
    if len(exp_results) < 2:
        state.pairwise_feedback = ""
        return
    sorted_by_score = sorted(exp_results, key=lambda result: ranking.adaptive_scores[id(result)], reverse=True)
    top_a, top_b = sorted_by_score[0], sorted_by_score[1]
    pairs_a = build_ref_gen_pairs(top_a)
    pairs_b = build_ref_gen_pairs(top_b)
    if not pairs_a or not pairs_b:
        state.pairwise_feedback = ""
        return

    rationale, score = await ctx.services.evaluation.pairwise_compare(pairs_a, pairs_b)

    winner = "A" if score > 0.5 else "B" if score < 0.5 else "TIE"
    logger.info("Pairwise: Exp %d vs Exp %d → %s (%s)", top_a.branch_id, top_b.branch_id, winner, rationale[:100])
    state.pairwise_feedback = (
        f"Top experiment {top_a.branch_id} vs runner-up {top_b.branch_id}: Winner={winner}. {rationale}"
    )


async def _run_independent_review(
    ranking: IterationRanking,
    proposals: list[ExperimentProposal],
    state: LoopState,
    ctx: RunContext,
) -> None:
    """Phase 3.9: independent review."""
    review = await review_iteration(
        experiments=ranking.exp_results,
        proposals=proposals,
        baseline_metrics=state.best_metrics,
        knowledge_base=state.knowledge_base,
        client=ctx.reasoning_client,
        model=ctx.config.reasoning_model,
    )
    state.review_feedback = review.strategic_guidance
    if review.strategic_guidance:
        logger.info("Review guidance: %.200s", review.strategic_guidance)


async def _run_replicate_gate(
    ranking: IterationRanking,
    state: LoopState,
    ctx: RunContext,
    iteration: int,
) -> None:
    """A1 paired-replicate gate orchestration.

    When ``config.replicates > 1``, replicate the top candidate + incumbent to populate
    ``ranking.best_replicate_scores`` / ``ranking.baseline_replicate_scores``; the policy layer
    (``_apply_iteration_result``) then uses ``replicate_promotion_decision`` instead of the
    single-shot epsilon check. On any candidate OR incumbent replication failure we skip
    silently — leaving ``baseline_replicate_scores = None`` so the policy layer falls back
    to single-shot. A half-populated state (candidate-only) would silently auto-promote via
    the empty-baseline branch of ``replicate_promotion_decision``, which is not what we want
    mid-run when an incumbent actually exists.
    """
    if ctx.config.replicates < 2:
        return

    tasks: list = [
        replicate_experiment(
            template=ranking.best_exp.template,
            branch_id=ranking.best_exp.branch_id,
            iteration=iteration,
            fixed_refs=state.fixed_references,
            config=ctx.config,
            n_replicates=ctx.config.replicates,
            existing_result=ranking.best_exp,
            services=ctx.services,
        )
    ]
    incumbent_template = state.best_template
    if incumbent_template is not None and incumbent_template.sections:
        tasks.append(
            replicate_experiment(
                template=incumbent_template,
                branch_id=_INCUMBENT_BRANCH_ID,
                iteration=iteration,
                fixed_refs=state.fixed_references,
                config=ctx.config,
                n_replicates=ctx.config.replicates,
                services=ctx.services,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    candidate_eval = results[0]
    incumbent_eval = results[1] if len(results) > 1 else None
    if isinstance(candidate_eval, BaseException):
        logger.warning(
            "Replicate gate: candidate replication failed (%s) — falling back to single-shot", candidate_eval
        )
        return

    # Incumbent-failure path must fall ALL the way back to single-shot: an empty baseline
    # would trip replicate_promotion_decision's "first iteration" branch (median > epsilon
    # only), silently auto-promoting mid-run candidates that haven't actually dominated.
    # Only populate replicate fields when BOTH replications succeed (or only candidate is
    # required when there's no incumbent to begin with).
    if isinstance(incumbent_eval, BaseException):
        logger.warning(
            "Replicate gate: incumbent replication failed (%s) — falling back to single-shot",
            incumbent_eval,
        )
        return

    # A1 ∘ A6: every scalar in ``replicate_promotion_decision`` must come from the same
    # protocol-appropriate scoring function — mixing composite and headroom scales across
    # replicates would corrupt the dominance check. Bind once.
    score = partial(_promotion_score, protocol=ctx.config.protocol)
    ranking.best_replicate_scores = [score(agg) for agg in candidate_eval.replicate_aggregated]
    ranking.best_exp.aggregated = candidate_eval.median_aggregated
    ranking.best_score = score(candidate_eval.median_aggregated)

    if incumbent_eval is None:
        # No incumbent template yet (first iteration) — replicate_promotion_decision treats
        # an empty baseline list as "use median-vs-epsilon only," which is the correct bootstrap.
        ranking.baseline_replicate_scores = []
    else:
        ranking.baseline_replicate_scores = [score(agg) for agg in incumbent_eval.replicate_aggregated]
        ranking.baseline_score = score(incumbent_eval.median_aggregated)


async def _synthesize_reasoning(
    ranking: IterationRanking,
    state: LoopState,
    ctx: RunContext,
) -> tuple[PromptTemplate, str] | None:
    """Phase 3.5a: reasoning call to merge top experiments into one template."""
    if len(ranking.exp_results) < 2:
        return None

    ranked_for_synth = sorted(ranking.exp_results, key=lambda result: ranking.adaptive_scores[id(result)], reverse=True)
    top_exps = ranked_for_synth[:3]
    logger.info("Synthesizing top %d experiments into merged template", len(top_exps))
    return await synthesize_templates(
        top_exps,
        state.style_profile,
        client=ctx.reasoning_client,
        model=ctx.config.reasoning_model,
        baseline_metrics=state.best_metrics,
    )


async def _run_synthesis_experiment(
    synth_template_result: tuple[PromptTemplate, str],
    ranking: IterationRanking,
    state: LoopState,
    ctx: RunContext,
    iteration: int,
) -> None:
    """Phase 3.5b: run the synthesis experiment and update ranking."""
    merged_template, merged_hypothesis = synth_template_result
    errors = validate_template(merged_template)
    if errors:
        logger.warning("Synthesis template invalid — skipping: %s", "; ".join(errors))
        return

    try:
        synth_result = await run_experiment(
            experiment_id=len(ranking.exp_results),
            template=merged_template,
            iteration=iteration,
            fixed_refs=state.fixed_references,
            config=ctx.config,
            last_results=state.last_iteration_results,
            hypothesis=merged_hypothesis,
            experiment_desc="Synthesis of top experiments",
            services=ctx.services,
        )
    except RuntimeError as exc:
        logger.warning("Synthesis experiment skipped: %s", exc)
        return
    merged_score = composite_score(synth_result.aggregated)
    logger.info(
        "Synthesis result: DS=%.4f Mega=%.4f (best individual: DS=%.4f Mega=%.4f)",
        synth_result.aggregated.dreamsim_similarity_mean,
        synth_result.aggregated.megastyle_similarity_mean,
        ranking.best_exp.aggregated.dreamsim_similarity_mean,
        ranking.best_exp.aggregated.megastyle_similarity_mean,
    )

    ranking.exp_results.append(synth_result)
    ranking.synth_result = synth_result
    updated_agg = [result.aggregated for result in ranking.exp_results]
    ranking.adaptive_scores = {
        id(result): adaptive_composite_score(result.aggregated, updated_agg) for result in ranking.exp_results
    }
    if merged_score > ranking.best_score:
        # Minimum-quality gate: synthesis can win the incumbent slot by averaging noise from two
        # sub-baseline parents. Require at least one parent to beat baseline on any perceptual /
        # canon axis before letting synthesis crown a new incumbent.
        baseline = state.best_metrics
        parent_beat_baseline = baseline is None
        if baseline is not None:
            for parent in ranking.exp_results:
                if parent is synth_result:
                    continue
                deltas = metric_deltas(parent.aggregated, baseline)
                if any(deltas.get(axis, 0.0) > IMPROVEMENT_EPSILON for axis in DELTA_METRIC_LABELS):
                    parent_beat_baseline = True
                    break
        if parent_beat_baseline:
            ranking.best_exp = synth_result
            ranking.best_score = merged_score
            logger.info("Synthesis beat best individual — adopting merged template")
        else:
            logger.info(
                "Synthesis beat best individual by composite, but no parent beat baseline on any single metric "
                "— holding synthesis out of promotion (synthesis visible to reasoner as analysis only)"
            )

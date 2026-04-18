"""Execution and analysis helpers for one iteration."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

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
    adaptive_composite_score,
    composite_score,
    improvement_epsilon,
    metric_deltas,
    paired_promotion_test,
)
from art_style_search.types import IterationResult, LoopState, PromotionTestResult, PromptTemplate
from art_style_search.utils import build_ref_gen_pairs
from art_style_search.workflow.context import RunContext
from art_style_search.workflow.policy import _candidate_results_for_validation

logger = logging.getLogger(__name__)

# Confirmatory-validation replicates use a dedicated branch_id outside the regular experiment range
# so they never collide with proposal indices in logs/state.
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
    promotion_test: PromotionTestResult | None = None
    best_replicate_scores: list[float] | None = None


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
            services=ctx.services,
        )
        for i, proposal in enumerate(proposals)
    ]
    return collect_experiment_results(await asyncio.gather(*exp_tasks, return_exceptions=True), "Experiment")


def _score_and_rank(exp_results: list[IterationResult], state: LoopState) -> IterationRanking:
    """Phase 3: compute aggregates, adaptive scores, pick best."""
    all_agg = [result.aggregated for result in exp_results]
    adaptive_scores = {id(result): adaptive_composite_score(result.aggregated, all_agg) for result in exp_results}
    best_exp = max(exp_results, key=lambda result: adaptive_scores[id(result)])
    best_score = composite_score(best_exp.aggregated)
    baseline_score = composite_score(state.best_metrics) if state.best_metrics else float("-inf")
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
    if state.silent_refs:
        feedback_set = frozenset(state.feedback_refs)
        pairs_a = [(ref, gen) for ref, gen in pairs_a if ref in feedback_set]
        pairs_b = [(ref, gen) for ref, gen in pairs_b if ref in feedback_set]
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


async def _confirmatory_validation(
    ranking: IterationRanking,
    state: LoopState,
    ctx: RunContext,
    iteration: int,
) -> None:
    """Phase 3.1: replicate top-2 candidates + incumbent for statistical testing."""
    if ctx.config.protocol != "rigorous":
        return

    candidates = _candidate_results_for_validation(ranking)
    if not candidates:
        return

    logger.info("Confirmatory validation: replicating %d candidates + incumbent", len(candidates))
    tasks = []
    for exp in candidates:
        tasks.append(
            replicate_experiment(
                template=exp.template,
                branch_id=exp.branch_id,
                iteration=iteration,
                fixed_refs=state.fixed_references,
                config=ctx.config,
                n_replicates=3,
                existing_result=exp,
                services=ctx.services,
            )
        )
    if state.best_template is not None:
        tasks.append(
            replicate_experiment(
                template=state.best_template,
                branch_id=_INCUMBENT_BRANCH_ID,
                iteration=iteration,
                fixed_refs=state.fixed_references,
                config=ctx.config,
                n_replicates=3,
                services=ctx.services,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    candidate_evals = []
    incumbent_eval = None
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning("Confirmatory replicate %d failed: %s", i, result)
            continue
        if i < len(candidates):
            candidate_evals.append((candidates[i], result))
        else:
            incumbent_eval = result

    if not candidate_evals or incumbent_eval is None:
        logger.warning("Confirmatory validation incomplete — falling back to single-pass scores")
        return

    for exp, evaluation in candidate_evals:
        exp.per_image_scores = list(evaluation.median_per_image)
        exp.aggregated = evaluation.median_aggregated

    updated_agg = [result.aggregated for result in ranking.exp_results]
    ranking.adaptive_scores = {
        id(result): adaptive_composite_score(result.aggregated, updated_agg) for result in ranking.exp_results
    }

    best_candidate_exp, best_candidate_eval = max(
        candidate_evals,
        key=lambda item: composite_score(item[1].median_aggregated),
    )
    ranking.best_exp = best_candidate_exp
    ranking.best_score = composite_score(best_candidate_exp.aggregated)
    ranking.best_replicate_scores = [composite_score(agg) for agg in best_candidate_eval.replicate_aggregated]
    test_result = paired_promotion_test(best_candidate_eval.median_per_image, incumbent_eval.median_per_image)
    logger.info(
        "Promotion test: p=%.4f, effect=%.5f, CI=[%.5f, %.5f], passed=%s",
        test_result.p_value,
        test_result.effect_size,
        test_result.ci_lower,
        test_result.ci_upper,
        test_result.passed,
    )
    ranking.promotion_test = test_result
    ranking.baseline_score = composite_score(incumbent_eval.median_aggregated)


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
        "Synthesis result: DS=%.4f (best individual: %.4f)",
        synth_result.aggregated.dreamsim_similarity_mean,
        ranking.best_exp.aggregated.dreamsim_similarity_mean,
    )

    ranking.exp_results.append(synth_result)
    ranking.synth_result = synth_result
    updated_agg = [result.aggregated for result in ranking.exp_results]
    ranking.adaptive_scores = {
        id(result): adaptive_composite_score(result.aggregated, updated_agg) for result in ranking.exp_results
    }
    if merged_score > ranking.best_score:
        # Minimum-quality gate: synthesis may merge sections from two sub-baseline parents and
        # win the incumbent slot by averaging noise. Require that at least one parent experiment
        # beat baseline on any canon-relevant or perceptual axis before letting synthesis crown
        # a new incumbent. Synthesis still runs + remains visible to the reasoner as analysis —
        # this only gates promotion.
        baseline = state.best_metrics
        parent_beat_baseline = False
        if baseline is None:
            parent_beat_baseline = True  # no baseline yet — nothing to guard against
        else:
            gate_axes = (
                "dreamsim_similarity_mean",
                "color_histogram_mean",
                "ssim_mean",
                "hps_score_mean",
                "aesthetics_score_mean",
                "vision_style",
                "vision_subject",
                "vision_composition",
                "vision_medium",
                "vision_proportions",
                "style_consistency",
            )
            for parent in ranking.exp_results:
                if parent is synth_result:
                    continue
                deltas = metric_deltas(parent.aggregated, baseline)
                if any(deltas.get(axis, 0.0) > 0.005 for axis in gate_axes):
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

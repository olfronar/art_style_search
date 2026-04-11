"""Per-iteration workflow helpers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from art_style_search.contracts import ExperimentProposal
from art_style_search.evaluate import pairwise_compare_experiments
from art_style_search.experiment import (
    best_kept_result,
    collect_experiment_results,
    replicate_experiment,
    run_experiment,
)
from art_style_search.knowledge import IterationDecision, build_caption_diffs, update_knowledge_base
from art_style_search.prompt import (
    Lessons,
    enforce_hypothesis_diversity,
    propose_experiments,
    review_iteration,
    synthesize_templates,
    validate_template,
)
from art_style_search.scoring import (
    adaptive_composite_score,
    composite_score,
    improvement_epsilon,
    paired_promotion_test,
)
from art_style_search.state import save_iteration_log, save_state
from art_style_search.types import AggregatedMetrics, IterationResult, LoopState, PromotionTestResult, PromptTemplate
from art_style_search.utils import build_ref_gen_pairs
from art_style_search.workflow.context import RunContext, _log_experiment_results, _save_best_prompt
from art_style_search.workflow.policy import _candidate_results_for_validation, _should_honor_stop

logger = logging.getLogger(__name__)

_MAX_PERSISTED_HISTORY = 30


def _filter_feedback_by_refs(feedback_text: str, feedback_refs: frozenset) -> str:
    """Filter multi-line per-image feedback to include only lines mentioning feedback_ref filenames."""
    if not feedback_text or not feedback_refs:
        return feedback_text
    ref_names = {path.name for path in feedback_refs}
    lines = feedback_text.split("\n")
    kept: list[str] = []
    keep_current = True
    for line in lines:
        if line.startswith("##") or not line.strip():
            kept.append(line)
            keep_current = True
            continue
        if line.startswith("**") or line.startswith("Image ("):
            keep_current = any(name in line for name in ref_names)
        if keep_current:
            kept.append(line)
    return "\n".join(kept)


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


def _build_iteration_context(state: LoopState) -> tuple[str, str, str]:
    """Phase 0 of an iteration: build (vision_fb, roundtrip_fb, caption_diffs)."""
    best_last = best_kept_result(state.last_iteration_results)
    vision_fb = best_last.vision_feedback if best_last else ""
    roundtrip_fb = best_last.roundtrip_feedback if best_last else ""

    feedback_set = frozenset(state.feedback_refs) if state.silent_refs else None
    if feedback_set and best_last:
        vision_fb = _filter_feedback_by_refs(vision_fb, feedback_set)
        roundtrip_fb = _filter_feedback_by_refs(roundtrip_fb, feedback_set)

    caption_diffs = ""
    if best_last and best_last.iteration_captions:
        captions_for_diff = best_last.iteration_captions
        scores_for_diff = best_last.per_image_scores
        if feedback_set:
            paired = [
                (caption, score)
                for caption, score in zip(captions_for_diff, scores_for_diff, strict=False)
                if caption.image_path in feedback_set
            ]
            captions_for_diff = [caption for caption, _ in paired]
            scores_for_diff = [score for _, score in paired]
        sorted_caps = sorted(
            zip(captions_for_diff, scores_for_diff, strict=False),
            key=lambda item: item[1].dreamsim_similarity,
        )
        worst_caps = [caption for caption, _ in sorted_caps[:3]]
        caption_diffs = build_caption_diffs(state.prev_best_captions, worst_caps)

    if state.review_feedback:
        roundtrip_fb = f"## Independent Review of Last Iteration\n{state.review_feedback}\n\n{roundtrip_fb}"
    if state.pairwise_feedback:
        vision_fb = f"## Pairwise Experiment Comparison\n{state.pairwise_feedback}\n\n{vision_fb}"

    state.review_feedback = ""
    state.pairwise_feedback = ""
    return vision_fb, roundtrip_fb, caption_diffs


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
                from art_style_search.types import ConvergenceReason

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
            from art_style_search.types import ConvergenceReason

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
            gemini_client=ctx.gemini_client,
            registry=ctx.registry,
            gemini_semaphore=ctx.gemini_semaphore,
            eval_semaphore=ctx.eval_semaphore,
            last_results=state.last_iteration_results,
            hypothesis=proposal.hypothesis,
            experiment_desc=proposal.experiment_desc,
            analysis=proposal.analysis,
            template_changes=proposal.template_changes,
            changed_section=proposal.changed_section,
            target_category=proposal.target_category,
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

    if ctx.services is None:
        rationale, score = await pairwise_compare_experiments(
            pairs_a,
            pairs_b,
            client=ctx.gemini_client,
            model=ctx.config.caption_model,
            semaphore=ctx.gemini_semaphore,
        )
    else:
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
                gemini_client=ctx.gemini_client,
                registry=ctx.registry,
                gemini_semaphore=ctx.gemini_semaphore,
                eval_semaphore=ctx.eval_semaphore,
                n_replicates=3,
                existing_scores=exp.per_image_scores,
                services=ctx.services,
            )
        )
    if state.best_template is not None:
        tasks.append(
            replicate_experiment(
                template=state.best_template,
                branch_id=900,
                iteration=iteration,
                fixed_refs=state.fixed_references,
                config=ctx.config,
                gemini_client=ctx.gemini_client,
                registry=ctx.registry,
                gemini_semaphore=ctx.gemini_semaphore,
                eval_semaphore=ctx.eval_semaphore,
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
    ranking.adaptive_scores = {id(result): adaptive_composite_score(result.aggregated, updated_agg) for result in ranking.exp_results}

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

    synth_result = await run_experiment(
        experiment_id=len(ranking.exp_results),
        template=merged_template,
        iteration=iteration,
        fixed_refs=state.fixed_references,
        config=ctx.config,
        gemini_client=ctx.gemini_client,
        registry=ctx.registry,
        gemini_semaphore=ctx.gemini_semaphore,
        eval_semaphore=ctx.eval_semaphore,
        last_results=state.last_iteration_results,
        hypothesis=merged_hypothesis,
        experiment_desc="Synthesis of top experiments",
        services=ctx.services,
    )
    merged_score = composite_score(synth_result.aggregated)
    logger.info(
        "Synthesis result: DS=%.4f (best individual: %.4f)",
        synth_result.aggregated.dreamsim_similarity_mean,
        ranking.best_exp.aggregated.dreamsim_similarity_mean,
    )

    ranking.exp_results.append(synth_result)
    ranking.synth_result = synth_result
    updated_agg = [result.aggregated for result in ranking.exp_results]
    ranking.adaptive_scores = {id(result): adaptive_composite_score(result.aggregated, updated_agg) for result in ranking.exp_results}
    if merged_score > ranking.best_score:
        ranking.best_exp = synth_result
        ranking.best_score = merged_score
        logger.info("Synthesis beat best individual — adopting merged template")


def _update_knowledge_base_for_iteration(
    state: LoopState,
    ranking: IterationRanking,
    proposals: list[ExperimentProposal],
    baseline_metrics: AggregatedMetrics | None,
    iteration: int,
    decision: IterationDecision = "rejected",
) -> None:
    """Phase 4 (KB): update the knowledge base after the iteration decision is known."""
    decision_by_id: dict[int, IterationDecision] = {result.branch_id: "rejected" for result in ranking.exp_results}
    selected = next((result for result in ranking.exp_results if result.kept), None)
    if selected is not None:
        decision_by_id[selected.branch_id] = decision

    for exp_result, proposal in zip(ranking.exp_results, proposals, strict=False):
        update_knowledge_base(
            state.knowledge_base,
            exp_result,
            exp_result.template,
            baseline_metrics,
            proposal,
            iteration,
            decision=decision_by_id[exp_result.branch_id],
        )

    if ranking.synth_result is not None:
        synth_result = ranking.synth_result
        synth_proposal = ExperimentProposal(
            template=synth_result.template,
            hypothesis=synth_result.hypothesis,
            experiment_desc=synth_result.experiment,
            builds_on=None,
            open_problems=[],
            lessons=Lessons(),
            target_category=synth_result.target_category,
        )
        update_knowledge_base(
            state.knowledge_base,
            synth_result,
            synth_result.template,
            baseline_metrics,
            synth_proposal,
            iteration,
            decision=decision_by_id[synth_result.branch_id],
        )


def _record_iteration_state(
    state: LoopState,
    ranking: IterationRanking,
    iteration: int,
    ctx: RunContext,
) -> None:
    """Persist iteration results and compact state."""
    current_best = best_kept_result(state.last_iteration_results)
    if current_best and current_best.iteration_captions:
        state.prev_best_captions = list(current_best.iteration_captions)

    state.last_iteration_results = ranking.exp_results
    state.experiment_history.extend(ranking.exp_results)
    if len(state.experiment_history) > _MAX_PERSISTED_HISTORY:
        state.experiment_history = state.experiment_history[-_MAX_PERSISTED_HISTORY:]

    _log_experiment_results(ranking.exp_results, ctx.config.log_dir, save_iteration_log)
    for result in ranking.exp_results:
        if not result.kept:
            result.iteration_captions = []
            result.rendered_prompt = ""
    save_state(state, ctx.config.state_file)
    _save_best_prompt(state, ctx.config.log_dir)

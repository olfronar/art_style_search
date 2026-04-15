"""Knowledge-base and state persistence helpers for iterations."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from art_style_search.contracts import ExperimentProposal, Lessons
from art_style_search.experiment import best_kept_result
from art_style_search.knowledge import IterationDecision, update_knowledge_base
from art_style_search.state import save_iteration_log, save_state
from art_style_search.types import AggregatedMetrics, LoopState
from art_style_search.workflow.context import RunContext, _log_experiment_results, _save_best_prompt

if TYPE_CHECKING:
    from art_style_search.workflow.iteration_execution import IterationRanking

# History entries are trimmed of captions+rendered prompt (both live in per-iteration
# JSON logs under {log_dir}/iter_NNN_branch_M.json).  10 entries ≈ one iteration of
# experiments, enough for the reasoning model to reference immediate prior results.
_MAX_PERSISTED_HISTORY = 10


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
    state.experiment_history.extend(replace(r, iteration_captions=[], rendered_prompt="") for r in ranking.exp_results)
    if len(state.experiment_history) > _MAX_PERSISTED_HISTORY:
        state.experiment_history = state.experiment_history[-_MAX_PERSISTED_HISTORY:]

    _log_experiment_results(ranking.exp_results, ctx.config.log_dir, save_iteration_log)
    for result in ranking.exp_results:
        if not result.kept:
            result.iteration_captions = []
            result.rendered_prompt = ""
    save_state(state, ctx.config.state_file)
    _save_best_prompt(state, ctx.config.log_dir)

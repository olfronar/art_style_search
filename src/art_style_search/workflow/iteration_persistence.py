"""Knowledge-base and state persistence helpers for iterations."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from art_style_search.contracts import ExperimentProposal, Lessons
from art_style_search.evaluate import extract_style_canon
from art_style_search.experiment import best_kept_result
from art_style_search.knowledge import (
    IterationDecision,
    append_kb_style_gap_observations,
    retire_resolved_style_gap_observations,
    update_knowledge_base,
)
from art_style_search.scoring import IMPROVEMENT_EPSILON, metric_deltas
from art_style_search.state import save_iteration_log, save_state
from art_style_search.types import AggregatedMetrics, CanonEditLedgerEntry, LoopState
from art_style_search.workflow.context import RunContext, _log_experiment_results, _save_best_prompt

if TYPE_CHECKING:
    from art_style_search.workflow.iteration_execution import IterationRanking

# History entries are trimmed of captions+rendered prompt (both live in per-iteration
# JSON logs under {log_dir}/iter_NNN_branch_M.json).  10 entries ≈ one iteration of
# experiments, enough for the reasoning model to reference immediate prior results.
_MAX_PERSISTED_HISTORY = 10

# Canon edit ledger ring-buffer capacity. Small by design — the reasoner only needs the
# last few edits to learn "I tried X → metric Y moved by Z". Older entries roll off.
_CANON_EDIT_LEDGER_MAX = 5
# Max canon excerpt length preserved per ledger entry; ~400 chars shows each side's shape.
_CANON_EDIT_EXCERPT_CHARS = 400
# Canon-affected metric axes surfaced in the ledger (style / medium / subject / consistency).
# MegaStyle is the primary continuous style-similarity signal (8% composite weight) — canon
# edits move it directly, so the ledger must surface its delta alongside the Gemini vision dims.
_CANON_LEDGER_METRIC_AXES: tuple[str, ...] = (
    "megastyle_similarity_mean",
    "vision_style",
    "vision_medium",
    "vision_subject",
    "vision_proportions",
    "style_consistency",
)
# Axes that count as "canon-relevant improvement" when retiring resolved style-gap observations.
# Observations come from the Gemini vision judge — they represent the judge's textual
# complaints. A MegaStyle improvement is a different signal entirely (continuous, image-space,
# anti-correlates with the ternary judge in practice — see the 2026-04 rebalance rationale),
# so it must NOT retire judge-sourced observations: the judge's concern is live until the
# judge's verdict changes. MegaStyle still appears in `_CANON_LEDGER_METRIC_AXES` so its
# delta shows on ledger entries; retirement gating stays on judge-accepting axes only.
_RETIRE_METRIC_AXES: tuple[str, ...] = (
    "vision_style",
    "vision_medium",
    "vision_subject",
    "style_consistency",
)


def _normalize(text: str) -> str:
    """Collapse internal whitespace for comparison + excerpt rendering."""
    return " ".join((text or "").split())


def _excerpt(text: str) -> str:
    """Return the first ``_CANON_EDIT_EXCERPT_CHARS`` of normalized text."""
    return _normalize(text)[:_CANON_EDIT_EXCERPT_CHARS]


_ACCEPTED_DECISIONS: frozenset[IterationDecision] = frozenset({"promoted", "exploration"})


def append_canon_edit_ledger(
    state: LoopState,
    ranking: IterationRanking,
    prior_canon: str,
    baseline_metrics: AggregatedMetrics | None,
    decision: IterationDecision,
    iteration: int,
) -> None:
    """Record this iteration's canon edit + measured effect in the ring buffer.

    Called after ``_apply_iteration_result`` so that *state.current_template* reflects
    the post-decision state. ``prior_canon`` is captured by the caller before the apply
    step — it's the canon the reasoner should see as "what I started from."

    Skips iterations where the canon is unchanged. Explicitly records rejected edits so
    the reasoner sees "I tried tightening Color Principle; it lost vision_subject."
    """
    if ranking.best_exp is None:
        return
    new_canon = extract_style_canon(ranking.best_exp.rendered_prompt or ranking.best_exp.template.render())
    if not new_canon or _normalize(new_canon) == _normalize(prior_canon):
        return
    deltas_raw: dict[str, float] = {}
    if baseline_metrics is not None:
        full_deltas = metric_deltas(ranking.best_exp.aggregated, baseline_metrics)
        deltas_raw = {axis: full_deltas[axis] for axis in _CANON_LEDGER_METRIC_AXES if axis in full_deltas}
    entry = CanonEditLedgerEntry(
        iteration=iteration,
        prior_canon_excerpt=_excerpt(prior_canon),
        new_canon_excerpt=_excerpt(new_canon),
        changed_sections=list(ranking.best_exp.changed_sections or []),
        hypothesis_summary=(ranking.best_exp.hypothesis or "")[:300],
        metric_deltas=deltas_raw,
        accepted=decision in _ACCEPTED_DECISIONS,
        canon_ops=list(ranking.best_exp.canon_ops or []),
    )
    state.canon_edit_ledger.append(entry)
    if len(state.canon_edit_ledger) > _CANON_EDIT_LEDGER_MAX:
        state.canon_edit_ledger = state.canon_edit_ledger[-_CANON_EDIT_LEDGER_MAX:]


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

    # Append the frontier experiment's style-gap observations (from the vision judge) to the KB
    # ring buffer so the next iteration's reasoner sees concrete canon-improvement feedstock.
    source = selected or (ranking.exp_results[0] if ranking.exp_results else None)
    if source is not None:
        append_kb_style_gap_observations(state.knowledge_base, source.aggregated.style_gap_notes)

    # Retire observations the selected experiment named AND resolved: (a) a lesson overlaps the
    # observation text and (b) at least one canon-relevant metric axis moved positive vs baseline.
    if selected is not None and baseline_metrics is not None:
        deltas = metric_deltas(selected.aggregated, baseline_metrics)
        canon_axes_improved = any(deltas.get(axis, 0.0) > IMPROVEMENT_EPSILON for axis in _RETIRE_METRIC_AXES)
        lesson_texts: list[str] = []
        if selected.hypothesis:
            lesson_texts.append(selected.hypothesis)
        for proposal in proposals:
            if proposal.template is selected.template and proposal.lessons:
                if proposal.lessons.new_insight:
                    lesson_texts.append(proposal.lessons.new_insight)
                if proposal.lessons.confirmed:
                    lesson_texts.append(proposal.lessons.confirmed)
        retire_resolved_style_gap_observations(
            state.knowledge_base,
            lesson_texts,
            canon_axes_improved,
        )

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

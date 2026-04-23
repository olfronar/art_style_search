"""Serialization and deserialization helpers for persisted state."""

from __future__ import annotations

import dataclasses
import enum
import json
from pathlib import Path
from typing import Any, get_args

from art_style_search.contracts import (
    ExperimentProposal,
    ExperimentSketch,
    Lessons,
    RefinementResult,
)
from art_style_search.types import (
    AggregatedMetrics,
    CanonEditLedgerEntry,
    Caption,
    CategoryProgress,
    ConvergenceReason,
    Hypothesis,
    IterationResult,
    KnowledgeBase,
    LoopState,
    MetricScores,
    OpenProblem,
    PromptSection,
    PromptTemplate,
    StyleProfile,
)
from art_style_search.workflow.proposal_recorder import (
    ProposalBatchRecorder,
    ProposalFate,
    ProposalRecord,
)


class _Encoder(json.JSONEncoder):
    """Custom JSON encoder for Path, dataclasses, and enums."""

    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return to_dict(o)
        if isinstance(o, enum.Enum):
            return o.value
        return super().default(o)


def to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass (or nested structure) to a JSON-safe dict."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            result[f.name] = to_dict(getattr(obj, f.name))
        return result
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, list):
        return [to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def _caption_from_dict(d: dict[str, Any]) -> Caption:
    return Caption(image_path=Path(d["image_path"]), text=d["text"])


def _metric_scores_from_dict(d: dict[str, Any]) -> MetricScores:
    return MetricScores(
        dreamsim_similarity=d.get("dreamsim_similarity", 0.0),
        hps_score=d["hps_score"],
        aesthetics_score=d["aesthetics_score"],
        color_histogram=d.get("color_histogram", 0.0),
        ssim=d.get("ssim", 0.0),
        vision_style=d.get("vision_style", 0.5),
        vision_subject=d.get("vision_subject", 0.5),
        vision_composition=d.get("vision_composition", 0.5),
        vision_medium=d.get("vision_medium", 0.5),
        vision_proportions=d.get("vision_proportions", 0.5),
        megastyle_similarity=d.get("megastyle_similarity", 0.0),
        style_gap=d.get("style_gap", ""),
        is_fallback=d.get("is_fallback", False),
    )


def _aggregated_metrics_from_dict(d: dict[str, Any]) -> AggregatedMetrics:
    return AggregatedMetrics(
        dreamsim_similarity_mean=d.get("dreamsim_similarity_mean", 0.0),
        dreamsim_similarity_std=d.get("dreamsim_similarity_std", 0.0),
        hps_score_mean=d["hps_score_mean"],
        hps_score_std=d["hps_score_std"],
        aesthetics_score_mean=d["aesthetics_score_mean"],
        aesthetics_score_std=d["aesthetics_score_std"],
        color_histogram_mean=d.get("color_histogram_mean", 0.0),
        color_histogram_std=d.get("color_histogram_std", 0.0),
        ssim_mean=d.get("ssim_mean", 0.0),
        ssim_std=d.get("ssim_std", 0.0),
        style_consistency=d.get("style_consistency", 0.0),
        vision_style=d.get("vision_style", 0.5),
        vision_subject=d.get("vision_subject", 0.5),
        vision_composition=d.get("vision_composition", 0.5),
        vision_style_std=d.get("vision_style_std", 0.0),
        vision_subject_std=d.get("vision_subject_std", 0.0),
        vision_composition_std=d.get("vision_composition_std", 0.0),
        vision_medium=d.get("vision_medium", 0.5),
        vision_medium_std=d.get("vision_medium_std", 0.0),
        vision_proportions=d.get("vision_proportions", 0.5),
        vision_proportions_std=d.get("vision_proportions_std", 0.0),
        megastyle_similarity_mean=d.get("megastyle_similarity_mean", 0.0),
        megastyle_similarity_std=d.get("megastyle_similarity_std", 0.0),
        completion_rate=d.get("completion_rate", 1.0),
        compliance_topic_coverage=d.get("compliance_topic_coverage", 1.0),
        compliance_marker_coverage=d.get("compliance_marker_coverage", 1.0),
        section_ordering_rate=d.get("section_ordering_rate", 1.0),
        section_balance_rate=d.get("section_balance_rate", 1.0),
        subject_specificity_rate=d.get("subject_specificity_rate", 1.0),
        style_canon_fidelity=d.get("style_canon_fidelity", 1.0),
        observation_boilerplate_purity=d.get("observation_boilerplate_purity", 1.0),
        style_gap_notes=tuple(d.get("style_gap_notes", [])),
        requested_ref_count=d.get("requested_ref_count", 0),
        actual_ref_count=d.get("actual_ref_count", 0),
    )


def _prompt_section_from_dict(d: dict[str, Any]) -> PromptSection:
    return PromptSection(name=d["name"], description=d["description"], value=d["value"])


def prompt_template_from_dict(d: dict[str, Any]) -> PromptTemplate:
    return PromptTemplate(
        sections=[_prompt_section_from_dict(s) for s in d.get("sections", [])],
        negative_prompt=d.get("negative_prompt"),
        caption_sections=d.get("caption_sections", []),
        caption_length_target=d.get("caption_length_target", 0),
    )


def style_profile_from_dict(d: dict[str, Any]) -> StyleProfile:
    return StyleProfile(
        color_palette=d["color_palette"],
        composition=d["composition"],
        technique=d["technique"],
        mood_atmosphere=d["mood_atmosphere"],
        subject_matter=d["subject_matter"],
        influences=d["influences"],
        gemini_raw_analysis=d["gemini_raw_analysis"],
        claude_raw_analysis=d["claude_raw_analysis"],
    )


def _iteration_result_from_dict(d: dict[str, Any]) -> IterationResult:
    changed_sections = d.get("changed_sections")
    if not changed_sections:
        changed_sections = [d["changed_section"]] if d.get("changed_section") else []
    return IterationResult(
        branch_id=d["branch_id"],
        iteration=d["iteration"],
        template=prompt_template_from_dict(d["template"]),
        rendered_prompt=d["rendered_prompt"],
        image_paths=[Path(p) for p in d["image_paths"]],
        per_image_scores=[_metric_scores_from_dict(s) for s in d["per_image_scores"]],
        aggregated=_aggregated_metrics_from_dict(d["aggregated"]),
        claude_analysis=d["claude_analysis"],
        template_changes=d["template_changes"],
        kept=d["kept"],
        hypothesis=d.get("hypothesis", ""),
        experiment=d.get("experiment", ""),
        vision_feedback=d.get("vision_feedback", ""),
        roundtrip_feedback=d.get("roundtrip_feedback", ""),
        iteration_captions=[_caption_from_dict(c) for c in d.get("iteration_captions", [])],
        n_images_attempted=d.get("n_images_attempted", 0),
        n_images_succeeded=d.get("n_images_succeeded", 0),
        changed_section=d.get("changed_section", ""),
        target_category=d.get("target_category", ""),
        changed_sections=changed_sections,
        direction_id=d.get("direction_id", ""),
        direction_summary=d.get("direction_summary", ""),
        failure_mechanism=d.get("failure_mechanism", ""),
        intervention_type=d.get("intervention_type", ""),
        risk_level=d.get("risk_level", "targeted"),
        expected_primary_metric=d.get("expected_primary_metric", ""),
        expected_tradeoff=d.get("expected_tradeoff", ""),
        canon_ops=list(d.get("canon_ops", [])),
    )


def _hypothesis_from_dict(d: dict[str, Any]) -> Hypothesis:
    return Hypothesis(
        id=d["id"],
        iteration=d["iteration"],
        parent_id=d.get("parent_id"),
        statement=d["statement"],
        experiment=d["experiment"],
        category=d["category"],
        outcome=d["outcome"],
        metric_delta=d.get("metric_delta", {}),
        kept=d["kept"],
        lesson=d.get("lesson", ""),
        direction_id=d.get("direction_id", ""),
        direction_summary=d.get("direction_summary", ""),
        failure_mechanism=d.get("failure_mechanism", ""),
        intervention_type=d.get("intervention_type", ""),
        risk_level=d.get("risk_level", "targeted"),
        expected_primary_metric=d.get("expected_primary_metric", ""),
        expected_tradeoff=d.get("expected_tradeoff", ""),
        changed_sections=d.get("changed_sections", []),
    )


def _open_problem_from_dict(d: dict[str, Any]) -> OpenProblem:
    return OpenProblem(
        text=d["text"],
        category=d["category"],
        priority=d["priority"],
        metric_gap=d.get("metric_gap"),
        since_iteration=d.get("since_iteration", 0),
    )


def _category_progress_from_dict(d: dict[str, Any]) -> CategoryProgress:
    return CategoryProgress(
        category=d["category"],
        best_perceptual_delta=d.get("best_perceptual_delta"),
        confirmed_insights=d.get("confirmed_insights", []),
        rejected_approaches=d.get("rejected_approaches", []),
        hypothesis_ids=d.get("hypothesis_ids", []),
        last_mechanism_tried=d.get("last_mechanism_tried", ""),
        last_confirmed_mechanism=d.get("last_confirmed_mechanism", ""),
    )


def _knowledge_base_from_dict(d: dict[str, Any]) -> KnowledgeBase:
    return KnowledgeBase(
        hypotheses=[_hypothesis_from_dict(h) for h in d.get("hypotheses", [])],
        categories={k: _category_progress_from_dict(v) for k, v in d.get("categories", {}).items()},
        open_problems=[_open_problem_from_dict(p) for p in d.get("open_problems", [])],
        style_gap_observations=list(d.get("style_gap_observations", [])),
        next_id=d.get("next_id", 1),
    )


def _loop_state_from_dict(d: dict[str, Any]) -> LoopState:
    return LoopState(
        iteration=d["iteration"],
        current_template=prompt_template_from_dict(d["current_template"]),
        best_template=prompt_template_from_dict(d["best_template"]),
        best_metrics=_aggregated_metrics_from_dict(d["best_metrics"]) if d.get("best_metrics") is not None else None,
        knowledge_base=_knowledge_base_from_dict(d["knowledge_base"]) if d.get("knowledge_base") else KnowledgeBase(),
        captions=[_caption_from_dict(c) for c in d["captions"]],
        style_profile=style_profile_from_dict(d["style_profile"]),
        fixed_references=[Path(p) for p in d.get("fixed_references", [])],
        experiment_history=[_iteration_result_from_dict(r) for r in d.get("experiment_history", [])],
        last_iteration_results=[_iteration_result_from_dict(r) for r in d.get("last_iteration_results", [])],
        prev_best_captions=[_caption_from_dict(c) for c in d.get("prev_best_captions", [])],
        plateau_counter=d.get("plateau_counter", 0),
        global_best_prompt=d.get("global_best_prompt", ""),
        global_best_metrics=(
            _aggregated_metrics_from_dict(d["global_best_metrics"])
            if d.get("global_best_metrics") is not None
            else None
        ),
        review_feedback=d.get("review_feedback", ""),
        pairwise_feedback=d.get("pairwise_feedback", ""),
        converged=d.get("converged", False),
        convergence_reason=(
            ConvergenceReason(d["convergence_reason"]) if d.get("convergence_reason") is not None else None
        ),
        seed=d.get("seed", 0),
        protocol=d.get("protocol", "short"),
        canon_edit_ledger=[_canon_edit_ledger_entry_from_dict(e) for e in d.get("canon_edit_ledger", [])],
    )


def _canon_edit_ledger_entry_from_dict(d: dict[str, Any]) -> CanonEditLedgerEntry:
    return CanonEditLedgerEntry(
        iteration=d.get("iteration", 0),
        prior_canon_excerpt=d.get("prior_canon_excerpt", ""),
        new_canon_excerpt=d.get("new_canon_excerpt", ""),
        changed_sections=list(d.get("changed_sections", [])),
        hypothesis_summary=d.get("hypothesis_summary", ""),
        metric_deltas=dict(d.get("metric_deltas", {})),
        accepted=bool(d.get("accepted", False)),
        canon_ops=list(d.get("canon_ops", [])),
    )


def _experiment_sketch_from_dict(d: dict[str, Any]) -> ExperimentSketch:
    return ExperimentSketch(
        hypothesis=d.get("hypothesis", ""),
        target_category=d.get("target_category", ""),
        failure_mechanism=d.get("failure_mechanism", ""),
        intervention_type=d.get("intervention_type", ""),
        direction_id=d.get("direction_id", ""),
        direction_summary=d.get("direction_summary", ""),
        risk_level=d.get("risk_level", "targeted"),
        expected_primary_metric=d.get("expected_primary_metric", ""),
        builds_on=d.get("builds_on", ""),
    )


def _lessons_from_dict(d: dict[str, Any]) -> Lessons:
    return Lessons(
        confirmed=d.get("confirmed", ""),
        rejected=d.get("rejected", ""),
        new_insight=d.get("new_insight", ""),
    )


def _refinement_result_from_dict(d: dict[str, Any]) -> RefinementResult:
    return RefinementResult(
        template=prompt_template_from_dict(d["template"]),
        analysis=d.get("analysis", ""),
        template_changes=d.get("template_changes", ""),
        should_stop=bool(d.get("should_stop", False)),
        hypothesis=d.get("hypothesis", ""),
        experiment=d.get("experiment", ""),
        lessons=_lessons_from_dict(d.get("lessons", {})),
        builds_on=d.get("builds_on"),
        open_problems=list(d.get("open_problems", [])),
        changed_section=d.get("changed_section", ""),
        changed_sections=d.get("changed_sections"),
        target_category=d.get("target_category", ""),
        direction_id=d.get("direction_id", ""),
        direction_summary=d.get("direction_summary", ""),
        failure_mechanism=d.get("failure_mechanism", ""),
        intervention_type=d.get("intervention_type", ""),
        risk_level=d.get("risk_level", "targeted"),
        expected_primary_metric=d.get("expected_primary_metric", ""),
        expected_tradeoff=d.get("expected_tradeoff", ""),
        canon_ops=list(d.get("canon_ops", [])),
    )


def _experiment_proposal_from_dict(d: dict[str, Any]) -> ExperimentProposal:
    return ExperimentProposal(
        template=prompt_template_from_dict(d["template"]),
        hypothesis=d.get("hypothesis", ""),
        experiment_desc=d.get("experiment_desc", ""),
        builds_on=d.get("builds_on"),
        open_problems=list(d.get("open_problems", [])),
        lessons=_lessons_from_dict(d.get("lessons", {})),
        analysis=d.get("analysis", ""),
        template_changes=d.get("template_changes", ""),
        changed_section=d.get("changed_section", ""),
        changed_sections=d.get("changed_sections"),
        target_category=d.get("target_category", ""),
        direction_id=d.get("direction_id", ""),
        direction_summary=d.get("direction_summary", ""),
        failure_mechanism=d.get("failure_mechanism", ""),
        intervention_type=d.get("intervention_type", ""),
        risk_level=d.get("risk_level", "targeted"),
        expected_primary_metric=d.get("expected_primary_metric", ""),
        expected_tradeoff=d.get("expected_tradeoff", ""),
        canon_ops=list(d.get("canon_ops", [])),
    )


_VALID_PROPOSAL_FATES: frozenset[str] = frozenset(get_args(ProposalFate))


def _proposal_record_to_dict(record: ProposalRecord) -> dict[str, Any]:
    """Serialize a ProposalRecord minus the transient ``refinement`` / ``proposal`` anchors.

    Those two fields exist only for in-iteration identity-map lookups (``refinement_to_rank`` /
    ``proposal_to_rank``) and are never consumed post-load — persisting them doubles the payload
    for no reader.
    """
    return {
        "rank": record.rank,
        "sketch": to_dict(record.sketch),
        "fate": record.fate,
        "fate_reason": record.fate_reason,
        "branch_id": record.branch_id,
    }


def proposal_batch_to_dict(recorder: ProposalBatchRecorder) -> dict[str, Any]:
    """Slim serializer for ``ProposalBatchRecorder`` that omits transient anchor fields."""
    return {
        "iteration": recorder.iteration,
        "records": [_proposal_record_to_dict(r) for r in recorder.records],
    }


def _proposal_record_from_dict(d: dict[str, Any]) -> ProposalRecord:
    fate_raw = d.get("fate", "brainstormed")
    fate: ProposalFate = fate_raw if fate_raw in _VALID_PROPOSAL_FATES else "brainstormed"
    refinement_raw = d.get("refinement")
    proposal_raw = d.get("proposal")
    return ProposalRecord(
        rank=int(d["rank"]),
        sketch=_experiment_sketch_from_dict(d["sketch"]),
        refinement=_refinement_result_from_dict(refinement_raw) if isinstance(refinement_raw, dict) else None,
        proposal=_experiment_proposal_from_dict(proposal_raw) if isinstance(proposal_raw, dict) else None,
        fate=fate,
        fate_reason=d.get("fate_reason"),
        branch_id=d.get("branch_id"),
    )


def proposal_batch_from_dict(d: dict[str, Any]) -> ProposalBatchRecorder:
    return ProposalBatchRecorder(
        iteration=int(d.get("iteration", 0)),
        records=[_proposal_record_from_dict(r) for r in d.get("records", [])],
    )

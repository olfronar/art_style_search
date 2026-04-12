"""Serialization and deserialization helpers for persisted state."""

from __future__ import annotations

import dataclasses
import enum
import json
from pathlib import Path
from typing import Any

from art_style_search.types import (
    AggregatedMetrics,
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
        completion_rate=d.get("completion_rate", 1.0),
        compliance_topic_coverage=d.get("compliance_topic_coverage", 1.0),
        compliance_marker_coverage=d.get("compliance_marker_coverage", 1.0),
        section_ordering_rate=d.get("section_ordering_rate", 1.0),
        section_balance_rate=d.get("section_balance_rate", 1.0),
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
    )


def _knowledge_base_from_dict(d: dict[str, Any]) -> KnowledgeBase:
    return KnowledgeBase(
        hypotheses=[_hypothesis_from_dict(h) for h in d.get("hypotheses", [])],
        categories={k: _category_progress_from_dict(v) for k, v in d.get("categories", {}).items()},
        open_problems=[_open_problem_from_dict(p) for p in d.get("open_problems", [])],
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
        protocol=d.get("protocol", "classic"),
        feedback_refs=[Path(p) for p in d.get("feedback_refs", [])],
        silent_refs=[Path(p) for p in d.get("silent_refs", [])],
    )
